/**
 * \file dnn/src/fallback/conv_bias/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "include/megdnn/thin/function.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/convolution/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"
#include "src/naive/conv_bias/opr_impl.h"

#include <unordered_map>

namespace megdnn {
namespace fallback {

/*!
 * \brief get the pack_size according to the format
 * Note  TODO: when remove format from param,
 *       may using like this "opr::param::format specify"
 * */
size_t pack_size(param::ConvBias::Format format);

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
    using AlgoDataType = detail::AlgoDataType;

    //! implemented by exec_with_ncb_kern()
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
              _megdnn_tensor_in bias, _megdnn_tensor_in z,
              _megdnn_tensor_out dst, const PreprocessedFilter*,
              _megdnn_workspace workspace) override;
    bool is_thread_safe() const override { return true; }

    void exec_preprocess(const TensorLayout& src_layout,
                         _megdnn_tensor_in filter,
                         _megdnn_tensor_in bias,
                         const TensorLayout& z_layout,
                         const TensorLayout& dst_layout,
                         PreprocessedFilter* preprocessed_filter,
                         _megdnn_workspace workspace) override;

    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) override;

    size_t get_preprocess_workspace_in_bytes(const TensorLayout& src,
                                             const TensorLayout& filter,
                                             const TensorLayout& bias,
                                             const TensorLayout& z,
                                             const TensorLayout& dst) override;

    //! implemented by get_workspace_with_ncb()
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& filter,
                                  const TensorLayout& bias,
                                  const TensorLayout& z,
                                  const TensorLayout& dst,
                                  const PreprocessedFilter*) override;

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
        //! when format is nchwxx, multi  channel will pack into one
        //! chnannel_pack_id. pack_channel_size is the number of packed channel
        //! when format is nchwxx and channel wise, multi group will pack into
        //! one group_pack_id. group_pack_size is the number of packed group
        //! together, like weight shape is {g/8, 1, 1, Fh, Fw, 8}
        template <typename T>
        const T* src(size_t batch_id, size_t group_pack_id,
                     size_t channel_pack_id = 0, size_t group_pack_size = 1,
                     size_t channel_pack_size = 1) const;

        template <typename T>
        const T* bias(size_t batch_id, size_t group_pack_id,
                      size_t channel_pack_id = 0, size_t group_pack_size = 1,
                      size_t channel_pack_size = 1) const;

        template <typename T>
        T* dst(size_t batch_id, size_t group_pack_id,
               size_t channel_pack_id = 0, size_t group_pack_size = 1,
               size_t channel_pack_size = 1) const;

        //! when format is nchwxx and channel wise, multi group will pack into
        //! one group_pack_id. group_pack_size is the number of packed group
        //! together, like weight shape is {g/8, 1, 1, Fh, Fw, 8}
        template <typename T>
        const T* filter(size_t group_pack_id,
                        size_t pack_group_size = 1_z) const;

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
            const ConvBiasImpl::NCBKernSizeParam& ncb_param) const {
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
        AlgoBase() : Algorithm() {
            m_handle_type = Handle::HandleType::FALLBACK;
        }

        enum class AlgoType : uint32_t {
            //! fallback
            FB_NAIVE = 1 << 0,
            FB_WINOGRAD_F32,
            FB_WINOGRAD_4X4_F32,
            FB_WINOGRAD_QS8,
            FB_WINOGRAD_8X8_QS8,
            FB_CONV1x1,
            FB_CONV1x1_GEMV,
            FB_IM2COL,

#if MEGDNN_X86
            X86_DIRECT = 1 << 8,
            X86_DIRECT_STRD2,
            X86_WINOGRAD_F63_8x8_F32,
            X86_WINOGRAD_F23_8x8_F32,
            X86_MKLDNN,
            X86_CHANWISE_AVX2_STRD1_QINT8,
            X86_CHANWISE_AVX2_STRD2_QINT8,
            X86_DIRECT_AVX2_STRD1_INT8,
            X86_DIRECT_AVX2_STRD2_INT8,
            X86_MKLDNN_QINT8,
            X86_MKLDNN_MATMUL_QINT8,
#elif MEGDNN_AARCH64 || MEGDNN_ARMV7
            ARM_COMMON_WINOGRAD_F23_FP16 = 1 << 8,
            ARM_COMMON_WINOGRAD_F45_FP16,
            ARM_COMMON_WINOGRAD_F63_FP16,
            ARM_COMMON_WINOGRAD_F23_8X8_FP16,
            ARM_COMMON_DIRECT_FP16,
            ARM_COMMON_DIRECT_STRD1_FP16,
            ARM_COMMON_WINOGRAD_F23_4X4_FP32,
            ARM_COMMON_WINOGRAD_F63_FP32,
            ARM_COMMON_WINOGRAD_F63_4X4_FP32,
            ARM_COMMON_WINOGRAD_F54_FP32,
            ARM_COMMON_WINOGRAD_F45_FP32,
            ARM_COMMON_WINOGRAD_F23_4X4_NCHW44_F32,
            ARM_COMMON_WINOGRAD_F63_4X4_NCHW44_F32,
            ARM_COMMON_WINOGRAD_F73_4X4_NCHW44_F32,
            ARM_COMMON_DIRECT_FP32,
            ARM_COMMON_DIRECT_STRD1_FP32,
            ARM_COMMON_DIRECT_STRD2_FP32,
            ARM_COMMON_DIRECT_NCHW44_FP32,
            ARM_COMMON_DIRECT_NCHW_NCHW44_FP32,
            ARM_COMMON_CHWNWISE_NCHW44_F32,
            ARM_COMMON_DIRECT_STRD1_S8,
            ARM_COMMON_DIRECT_STRD2_S8,
            ARM_COMMON_DIRECT_NCHW44,
            ARM_COMMON_DIRECT_NCHW_NCHW44_S8,
            ARM_COMMON_CHANWISE_STRD1_NCHW44_S8,
            ARM_COMMON_CHANWISE_STRD2_NCHW44_S8,
            ARM_COMMON_DIRECT_NCHW_NCHW44_DOT_S8,
            ARM_COMMON_DIRECT_STRD1_DOT_S8,
            ARM_COMMON_DIRECT_STRD2_DOT_S8,
            ARM_COMMON_DIRECT_NCHW44_DOT_S8,
            ARM_COMMON_WINOGRAD_F23_8X8_S8,
            ARM_COMMON_WINOGRAD_F23_8X8_NCHW44_S8CF32,
            ARM_COMMON_WINOGRAD_F23_8X8_NCHW44_S8,
            ARM_COMMON_DIRECT_INT8X8X16,
            ARM_COMMON_DIRECT_NCHW44_INT8X8X16,
            ARM_COMMON_DIRECT_STRD2_INT8X8X16,
            ARM_COMMON_DIRECT_STRD2_F2_INT8X8X16,
            ARM_COMMON_CHWNWISE_STRD1_STRD2_NCHW44_INT8X8X16,
            ARM_COMMON_DIRECT_NCHW_NCHW44_INT8X8X16,
            ARM_COMMON_DIRECT_STRD1_QU8,
            ARM_COMMON_DIRECT_STRD2_QU8,
            ARM_COMMON_DIRECT_STRD1_DOT_QU8,
            ARM_COMMON_DIRECT_STRD2_DOT_QU8,
#if MEGDNN_AARCH64
            AARCH64_DIRECT_STRD2_FP16,
            AARCH64_DIRECT_STRD2_FP32,
            AARCH64_MATMUL_S8,
            AARCH64_MATMUL_QU8,
#else
            ARMV7_MATMUL_S8,
            ARMV7_MATMUL_QU8,
#endif // MEGDNN_AARCH64
#endif
        };

        virtual ~AlgoBase() = default;
        virtual bool usable(
                const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const = 0;
        virtual size_t get_workspace(const NCBKernSizeParam& param) const = 0;

        virtual SmallVector<NCBKern> dispatch_kerns(
                const NCBKernSizeParam& param) const = 0;

        virtual SmallVector<NCBKern> dispatch_preprocess_kerns(
                const NCBKernSizeParam&) const {
            return {};
        };

        //! get the layouts of weight_prerocess dst
        virtual SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
                const NCBKernSizeParam&) const {
            return {};
        };

        //! get the workspace when weight_prerocess
        virtual size_t get_preprocess_workspace(const NCBKernSizeParam&) const {
            return 0_z;
        };

        //! Temporarily used to identify whether the matmul algorithm is
        //! is_preferred.
        virtual bool is_preferred(const NCBKernSizeParam&) const {
            return false;
        }
        bool usable_reproducible(const NCBKernSizeParam& param,
                                 AlgoSelectionStrategy algo_selection_strategy,
                                 bool reproducible = true) const {
            return (!reproducible || is_reproducible()) &&
                   usable(param, algo_selection_strategy);
        }

        //! get the type of the algo
        virtual ConvAlgoTypePack get_algo_type() const = 0;
        using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;
    };

    using AlgoMapper = AlgoBase::Mapper;
    /**
     * \brief get all the algorithm for the opr.
     */
    virtual SmallVector<AlgoBase*> get_all_packed_algo();

    /**
     * \brief select algo according to input algo type
     */
    SmallVector<AlgoBase*> select_algo_type(ConvAlgoTypePack algo_type);

    /**
     * \brief suggest algo category according to the param
     */
    virtual SmallVector<AlgoCategory> suggest_algo_category_order(
            const NCBKernSizeParam& param) const;

protected:
    virtual void exec_with_ncb_kern(const NCBKernParam& param,
                                    ConvBiasImpl::Algorithm* algo);

    virtual void exec_preprocess_with_ncb_kern(const NCBKernParam& param,
                                               Algorithm* algo);

    virtual std::vector<Algorithm*> get_all_algorithms_with_ncb(
            const NCBKernSizeParam& param);

    virtual Algorithm* get_algorithm_heuristic_with_ncb(
            const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
            bool reproducible = false);

    const char* get_algorithm_set_name() const override;

private:
    class AlgoNaive;
    class AlgoIm2col;
    class AlgoConv1x1;
    class AlgoConv1x1Gemv;
    class AlgoWinogradF32;
    class AlgoWinogradF32_4x4;
    class AlgoWinogradQS8;
    class AlgoWinogradQS8_8x8;
    class AlgoPack;

    NCBKernSizeParam m_prev_selected_algo_sizep;
    Algorithm* m_prev_selected_algo = nullptr;

    bool is_naive_algo(ConvBiasImpl::Algorithm* algo);

    Algorithm* get_algo_from_desc(const AlgorithmDesc& desc) const;

    //! get algorithm set by user or by heuristic
    Algorithm* get_algorithm(
            const NCBKernSizeParam& param,
            size_t workspace_size = std::numeric_limits<size_t>::max());

    NCBKernSizeParam make_ncb_kern_size_param(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& dst,
            const PreprocessedFilter* preprocessed_filter);

    NCBKernParam make_ncb_kern_param(
            _megdnn_tensor_in src, _megdnn_tensor_in filter,
            _megdnn_tensor_in bias, _megdnn_tensor_out dst,
            _megdnn_workspace workspace,
            const PreprocessedFilter* preprocessed_filter);

    static const AlgoPack& algo_pack();
};

inline bool is_enable_filter_preprocess(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    return param.preprocessed_filter &&
           param.preprocessed_filter->tensors.size() >= 1;
}
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
