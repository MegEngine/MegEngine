/**
 * \file dnn/src/fallback/convolution/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <memory>
#include <unordered_map>
#include "megdnn/oprs/base.h"
#include "src/common/utils.h"
#include "src/common/algo_base.h"
#include "src/fallback/handle.h"
#include "src/naive/convolution/opr_impl.h"

namespace megdnn {

/**
 * \brief Convolutino algo category
 */
enum class AlgoCategory : int32_t {
    DIRECT = 0,
    IM2COL = 1,
    WINOGRAD = 2,
    NAIVE = 3,
};

struct ConvAlgoTypePack {
    detail::AlgoDataType data_type : 32;
    AlgoCategory algo_category : 32;
};

namespace fallback {

/*!
 * \brief fallback convolution forward impl
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
class ConvolutionImpl : public naive::ConvolutionForwardImpl {
public:
    using naive::ConvolutionForwardImpl::ConvolutionForwardImpl;
    using AlgoSelectionStrategy = detail::AlgoSelectionStrategy;
    using AlgoDataType = detail::AlgoDataType;

    //! implemented by exec_with_ncb_kern()
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
              _megdnn_tensor_out dst, const PreprocessedFilter*,
              _megdnn_workspace workspace) override;

    void exec_preprocess(const TensorLayout& src_layout,
                         _megdnn_tensor_in filter,
                         const TensorLayout& dst_layout,
                         PreprocessedFilter* preprocessed_filter,
                         _megdnn_workspace workspace) override;

    //! implemented by get_workspace_with_ncb()
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& filter,
                                  const TensorLayout& dst,
                                  const PreprocessedFilter*) override;

    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;

    size_t get_preprocess_workspace_in_bytes(const TensorLayout& src,
                                             const TensorLayout& filter,
                                             const TensorLayout& dst) override;

    //! implemented by get_all_algorithms_with_ncb()
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;

    //! implemented by get_algorithm_heuristic_with_ncb()
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& filter,
                                       const TensorLayout& dst,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;

    //! size param for kernels with non-contiguous batch
    struct NCBKernSizeParam {
        uint32_t n;
        std::array<uint32_t, MAX_SPATIAL_DIM> isz, osz;
        //! filter info; group is guaranteed to be 1
        CanonizedFilterMeta filter_meta;
        DType src_type, filter_type, dst_type;
        //! stride for batch of input, output
        ptrdiff_t inp_bs, out_bs;
        //! stride for each dim of input, output
        ptrdiff_t inp_s[4], out_s[4];
        Param::ComputeMode compute_mode;
        size_t nr_threads;
        //! weight_preprocess info
        const PreprocessedFilter* preprocessed_filter;
        //! get the data type category of the param for select the algo
        AlgoDataType deduce_algo_data_type() const;
    };

    //! memory param for kernels with non-contiguous batch
    struct NCBKernParam : public NCBKernSizeParam {
        const void* src_ptr;
        const void* filter_ptr;
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
        T* dst() const {
            dst_type.assert_is_compatible_ctype<T>();
            return static_cast<T*>(dst_ptr);
        }

        template <typename T>
        T* workspace() const {
            return static_cast<T*>(workspace_ptr);
        }

        //! when format is nchwxx and channel wise, multi group will pack into
        //! one group_pack_id. group_pack_size is the number of packed group
        //! together, like weight shape is {g/8, 1, 1, Fh, Fw, 8}
        template <typename T>
        T* dst(size_t batch_id, size_t group_pack_id,
               size_t group_pack_size = 1_z) const{
            size_t batch_offset = batch_id * out_bs * dst_type.size();
            size_t group_offset = group_pack_size * group_pack_id *
                                  filter_meta.ocpg * osz[0] * osz[1] *
                                  dst_type.size();
            return reinterpret_cast<T*>(reinterpret_cast<ptrdiff_t>(dst_ptr) +
                                        batch_offset + group_offset);
        }

        template <typename T>
        const T* src(size_t batch_id, size_t group_pack_id,
                     size_t group_pack_size = 1_z) const {
            size_t batch_offset = batch_id * inp_bs * src_type.size();
            size_t group_offset = group_pack_size * group_pack_id *
                                  filter_meta.icpg * isz[0] * isz[1] *
                                  src_type.size();
            return reinterpret_cast<T*>(reinterpret_cast<ptrdiff_t>(src_ptr) +
                                        batch_offset + group_offset);

        }

        template <typename T>
        const T* filter(size_t group_pack_id,
                        size_t pack_group_size = 1_z) const {
            size_t group_offset = pack_group_size * group_pack_id *
                                  filter_meta.icpg * filter_meta.ocpg *
                                  filter_meta.spatial[0] *
                                  filter_meta.spatial[1] * filter_type.size();
            return reinterpret_cast<T*>(
                    reinterpret_cast<ptrdiff_t>(filter_ptr) + group_offset);
        }
    };

    /**
     * \brief Kernel run time id, This information is used for getting the
     * work data
     */
    struct NCBKernIndex {
        size_t thread_id = 0;  //!< Thread id
        CpuNDRange ndrange_id;
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
            FB_ALGO = 1 << 0,
            FB_NAIVE,
            FB_DEFAULT,
        };

        virtual ~AlgoBase() = default;
        virtual bool usable(const NCBKernSizeParam& param,
                            AlgoSelectionStrategy) const = 0;
        virtual size_t get_workspace(const NCBKernSizeParam& param) const = 0;
        virtual SmallVector<NCBKern> dispatch_kern(
                const NCBKernSizeParam& param) const = 0;

        virtual SmallVector<NCBKern> dispatch_preprocess_kern(
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

    /**
     * \brief get all the algorithm for the opr.
     */
    virtual SmallVector<AlgoBase*> get_all_packed_algo();

    /**
     * \brief select algo according to input algo type
     */
    SmallVector<AlgoBase*> select_algo_type(ConvAlgoTypePack algo_type);

protected:
    virtual void exec_with_ncb_kern(const NCBKernParam& param, Algorithm* algo);

    virtual void exec_preprocess_with_ncb_kern(const NCBKernParam& param,
                                               Algorithm* algo);

    virtual std::vector<Algorithm*> get_all_algorithms_with_ncb(
            const NCBKernSizeParam& param);

    virtual Algorithm* get_algorithm_heuristic_with_ncb(
            const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
            bool reproducible = false);

    const char* get_algorithm_set_name() const override;

    class AlgoFallback;
    class AlgoNaive;
    class AlgoDefault;
    class AlgoPack;

private:

    NCBKernSizeParam m_prev_selected_algo_sizep;
    Algorithm* m_prev_selected_algo = nullptr;

    Algorithm* get_algo_from_desc(const AlgorithmDesc& desc) const;
    bool is_naive_algo(ConvolutionImpl::Algorithm* algo);
    Algorithm* get_algorithm(
            const NCBKernSizeParam& param,
            size_t workspace_size = std::numeric_limits<size_t>::max());

    NCBKernSizeParam make_ncb_kern_size_param(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst,
            const PreprocessedFilter* preprocessed_filter);

    NCBKernParam make_ncb_kern_param(
            _megdnn_tensor_in src, _megdnn_tensor_in filter,
            _megdnn_tensor_out dst,
            const PreprocessedFilter* preprocessed_filter,
            _megdnn_workspace workspace);

    SmallVector<AlgoCategory> suggest_algo_category_order(
            const NCBKernSizeParam& param) const;

public:
    static const AlgoPack& algo_pack();
};

class ConvolutionBackwardDataImpl : public naive::ConvolutionBackwardDataImpl {
public:
    using naive::ConvolutionBackwardDataImpl::ConvolutionBackwardDataImpl;

    void exec(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& flter,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override;
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& filter,
                                       const TensorLayout& diff,
                                       const TensorLayout& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;
    const char* get_algorithm_set_name() const override;

    //! size param for kernels with non-contiguous batch
    struct NCBKernSizeParam {
        uint32_t n;
        std::array<uint32_t, MAX_SPATIAL_DIM> isz, osz;
        //! filter info; group is guaranteed to be 1
        CanonizedFilterMeta filter_meta;
        DType diff_type, filter_type, grad_type;
        TensorLayout diff_layout, filter_layout, grad_layout;
        //! stride for batch of input, output
        ptrdiff_t inp_bs, out_bs;
        //! extra_mem_size (in bytes) memory after the end of the logical
        //! memory block is accessible.
        //!
        //! this allows for eliminating unnecessary memory copies: e.g.
        //! if several bytes after the end of the tensor are
        //! accessible, some kernel implementations can utilize
        //! out-of-bound SIMD memory access, to avoid issuing
        //! memcpy instructions.
        //!
        //! Note that although extra_mem_size bytes are accessible by the
        //! kernel implementation, kernel implementation should not have any
        //! ``visible'' effect on any unintended memory location.
        //! This means reading and writing the same value to some memory
        //! location within extra_mem_size is allowed, but writing a
        //! different value is not allowed.
        size_t diff_extra_mem_size, filter_extra_mem_size, grad_extra_mem_size;
        Param::ComputeMode compute_mode;
    };

    //! memory param for kernels with non-contiguous batch
    struct NCBKernParam : public NCBKernSizeParam {
        const void* filter_ptr;
        const void* diff_ptr;
        void* grad_ptr;
        void* workspace_ptr;
        size_t workspace_size;

        template <typename T>
        const T* diff() const {
            diff_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(diff_ptr);
        }

        template <typename T>
        const T* filter() const {
            filter_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(filter_ptr);
        }

        template <typename T>
        T* grad() const {
            grad_type.assert_is_compatible_ctype<T>();
            return static_cast<T*>(grad_ptr);
        }

        template <typename T>
        T* workspace() const {
            return static_cast<T*>(workspace_ptr);
        }
    };

protected:
    using ncb_kern_t = thin_function<void(const NCBKernParam& param)>;
    class AlgoBase : public Algorithm {
    protected:
        ~AlgoBase() = default;

    public:
        AlgoBase() : Algorithm() {
            m_handle_type = Handle::HandleType::FALLBACK;
        }
        enum class AlgoType : uint32_t {
            //! fallback
            FB_NAIVE = 1 << 0,
            FB_DIRECT,
            FB_MATMUL,

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
            ARM_COMMON_DIRECT_STRD1_DOT_INT8X8X32 = 1 << 8,
            ARM_COMMON_DIRECT_STRD2_DOT_INT8X8X32,
            ARM_COMMON_DIRECT_STRD1_DOT_QU8,
            ARM_COMMON_DIRECT_STRD2_DOT_QU8
#endif
        };

        virtual bool usable(ConvolutionBackwardDataImpl* opr,
                            const NCBKernSizeParam& param) const = 0;
        virtual size_t get_workspace(ConvolutionBackwardDataImpl* opr,
                                     const NCBKernSizeParam& param) const = 0;
        virtual ncb_kern_t dispatch_kern(
                ConvolutionBackwardDataImpl* opr,
                const NCBKernSizeParam& param) const = 0;
        bool usable_reproducible(ConvolutionBackwardDataImpl* opr,
                                 const NCBKernSizeParam& param,
                                 bool reproducible = true) const {
            return (!reproducible || is_reproducible()) && usable(opr, param);
        }
        virtual bool is_preferred(const NCBKernSizeParam&) const {
            return false;
        }
        //! if the algo is naive, it will not split by group
        virtual bool is_naive() const { return false; }
        using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;
    };

protected:

    //! default impl calls ncb_1g_dispatch_kern()
    virtual void exec_with_ncb_kern(const NCBKernParam& param);

    //! default impl calls ncb_1g_get_workspace()
    virtual size_t get_workspace_with_ncb(const NCBKernSizeParam& param);

    //! default impl calls ncb_1g_get_all_algorithms()
    virtual std::vector<Algorithm*> get_all_algorithms_with_ncb(
            const NCBKernSizeParam& param);

    //! default impl calls ncb_1g_get_algorithm_heuristic()
    virtual Algorithm* get_algorithm_heuristic_with_ncb(
            const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
            bool reproducible = false);

    //! get kernel pointer for float32 non-contiguous batch 1-group kernel
    virtual ncb_kern_t ncb_1g_dispatch_kern(Algorithm* algo,
                                            const NCBKernSizeParam& param);

    virtual size_t ncb_1g_get_workspace(Algorithm* algo,
                                        const NCBKernSizeParam& param);

    virtual std::vector<Algorithm*> ncb_1g_get_all_algorithms(
            const NCBKernSizeParam& param);

    /*!
     * the default impl iterates over all ncb_1g_get_all_algorithms()
     * and return the first one whose workspace does not exceed the limit.
     */
    virtual Algorithm* ncb_1g_get_algorithm_heuristic(
            const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
            bool reproducible = false);

    static bool is_matrix_mul_preferred(const NCBKernSizeParam& param);
    /**
     * \brief get all the algorithm for the opr.
     */
    virtual SmallVector<AlgoBase*> get_all_packed_algo();

private:
    NCBKernSizeParam m_prev_selected_algo_sizep;
    Algorithm* m_prev_selected_algo = nullptr;

    //! get algorithm set by user or by heuristic
    Algorithm* get_algorithm(const NCBKernSizeParam& param);

    NCBKernSizeParam make_ncb_kern_size_param(const TensorLayout& filter,
                                              const TensorLayout& diff,
                                              const TensorLayout& grad);

    NCBKernParam make_ncb_kern_param(_megdnn_tensor_in filter,
                                     _megdnn_tensor_in diff,
                                     _megdnn_tensor_out grad,
                                     _megdnn_workspace workspace);

    class AlgoNaive;
    class AlgoDirect;
    class AlgoMatrixMul;
    class AlgoPack;
    Algorithm* get_algo_from_desc(const AlgorithmDesc& desc) const;

public:
    //! maintain all the algos of in the opr of fallback
    static const AlgoPack& algo_pack();
};

}  // namespace fallback
}  // namespace megdnn

//! unpack NCBKernSizeParam into local variables (N, IC, IH, IW, ...)
#define UNPACK_CONV_F32_NCB_KERN_SIZES(_p)                                   \
    auto N = _p.n, IC = _p.filter_meta.icpg, IH = _p.isz[0], IW = _p.isz[1], \
         OC = _p.filter_meta.ocpg, OH = _p.osz[0], OW = _p.osz[1],           \
         FH = _p.filter_meta.spatial[0], FW = _p.filter_meta.spatial[1],     \
         SH = _p.filter_meta.stride[0], SW = _p.filter_meta.stride[1],       \
         PH = _p.filter_meta.padding[0], PW = _p.filter_meta.padding[1]

// vim: syntax=cpp.doxygen
