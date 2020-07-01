/**
 * \file dnn/src/naive/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/basic_types.h"

#include "src/common/handle_impl.h"
#include "src/naive/convolution/algorithms.h"
#include "src/naive/local_share/algorithms.h"
#include "src/naive/convolution3d/algorithms.h"

#include <functional>
#include <mutex>
#include <type_traits>

namespace megdnn {
namespace naive {

class HandleImpl : public HandleImplHelper {
    using KernFunc = MegcoreCPUDispatcher::Task;
    using MultiThreadingKernFunc = MegcoreCPUDispatcher::MultiThreadingTask;
    MegcoreCPUDispatcher* m_dispatcher;

    static DefaultConvolutionForwardAlgorithm m_default_conv_fwd_algo;
    static DefaultConvolutionBackwardDataAlgorithm m_default_conv_bwd_data_algo;
    static DefaultConvolutionBackwardFilterAlgorithm
            m_default_conv_bwd_filter_algo;
    static DefaultConvBiasForwardAlgorithm m_default_conv_bias_fwd_algo;
    static DefaultConvolution3DForwardAlgorithm m_default_conv3d_fwd_algo;
    static DefaultConvolution3DBackwardDataAlgorithm
            m_default_conv3d_bwd_data_algo;
    static DefaultConvolution3DBackwardFilterAlgorithm
            m_default_conv3d_bwd_filter_algo;
    static DefaultBatchConvBiasForwardAlgorithm
            m_default_batch_conv_bias_fwd_algo;
    static DefaultLocalShareForwardAlgorithm m_default_local_share_fwd_algo;
    static DefaultLocalShareBackwardDataAlgorithm
            m_default_local_share_bwd_data_algo;
    static DefaultLocalShareBackwardFilterAlgorithm
            m_default_local_share_bwd_filter_algo;

    //! move KernFunc to alloc_kern()->func, destruct func, and call dispatch
    template <typename T>
    void move_kern_func_to_new_kern_and_dispatch(T& func) {
        m_dispatcher->dispatch(std::move(func));
        func.~T();
    }

    template <typename T>
    void move_kern_func_to_new_kern_and_dispatch(T& func, size_t parallelism) {
        m_dispatcher->dispatch(std::move(func), parallelism);
        func.~T();
    }

public:
    HandleImpl(megcoreComputingHandle_t computing_handle,
               HandleType type = HandleType::NAIVE);

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();

    ConvolutionForward::Algorithm* default_conv_fwd_algo() {
        return &m_default_conv_fwd_algo;
    }

    ConvolutionBackwardData::Algorithm* default_conv_bwd_data_algo() {
        return &m_default_conv_bwd_data_algo;
    }

    ConvolutionBackwardFilter::Algorithm* default_conv_bwd_filter_algo() {
        return &m_default_conv_bwd_filter_algo;
    }

    ConvBiasForward::Algorithm* default_conv_bias_fwd_algo() {
        return &m_default_conv_bias_fwd_algo;
    }

    Convolution3DForward::Algorithm* default_conv3d_fwd_algo() {
        return &m_default_conv3d_fwd_algo;
    }
    Convolution3DBackwardData::Algorithm* default_conv3d_bwd_data_algo() {
        return &m_default_conv3d_bwd_data_algo;
    }

    Convolution3DBackwardFilter::Algorithm* default_conv3d_bwd_filter_algo() {
        return &m_default_conv3d_bwd_filter_algo;
    }

    BatchConvBiasForward::Algorithm* default_batch_conv_bias_fwd_algo() {
        return &m_default_batch_conv_bias_fwd_algo;
    }

    LocalShareForward::Algorithm* default_local_share_fwd_algo() {
        return &m_default_local_share_fwd_algo;
    }

    LocalShareBackwardData::Algorithm* default_local_share_bwd_data_algo() {
        return &m_default_local_share_bwd_data_algo;
    }

    LocalShareBackwardFilter::Algorithm* default_local_share_bwd_filter_algo() {
        return &m_default_local_share_bwd_filter_algo;
    }

    Relayout* relayout_opr() override {
        return get_helper_opr<Relayout, 2>(this);
    }
    /*!
     * \brief pass a kernel to the dispatcher associated with the megcore
     *      computing handle
     */
    template <class T>
    void dispatch_kern(T&& kern) {
        // this impl mainly serves to reduce binary size: we only need to
        // call ctor here, and dtor can be called from the cpp so its code
        // only needs to be generated once
        std::aligned_storage<sizeof(KernFunc), alignof(KernFunc)>::type s;
        move_kern_func_to_new_kern_and_dispatch(
                *new (&s) KernFunc(std::forward<T>(kern)));
    }

    /*!
     * \brief pass a kernel to the multi thread dispatcher associated with the
     * megcore computing handle
     */
    template <class T>
    void dispatch_kern(T&& kern, size_t parallelism) {
        // this impl mainly serves to reduce binary size: we only need to
        // call ctor here, and dtor can be called from the cpp so its code
        // only needs to be generated once
        std::aligned_storage<sizeof(MultiThreadingKernFunc),
                             alignof(MultiThreadingKernFunc)>::type s;
        move_kern_func_to_new_kern_and_dispatch(
                *new (&s) MultiThreadingKernFunc(std::forward<T>(kern)),
                parallelism);
    }

    MegcoreCPUDispatcher* megcore_dispatcher() const { return m_dispatcher; }

    //! note: the impl requires the handle type to be exactly NAIVE
    size_t image2d_pitch_alignment() const override;

    /*!
     * \brief set the value of image2d_pitch_alignment() and return original
     *      setting
     *
     * This is only used in test cases where we need to use a naive impl on
     * specific tensor format.
     *
     * \param alignment the new alignment value to set
     */
    static size_t exchange_image2d_pitch_alignment(size_t alignment);
};

}  // namespace naive
}  // namespace megdnn

/*!
 * \brief operator impls should utilize this method to
 * \param _handle a pointer to HandleImpl
 * \param _stmt the statements to be executed for the kernel
 */
#define MEGDNN_DISPATCH_CPU_KERN(_handle, _stmt) \
    do {                                         \
        auto _kern = [=]() { _stmt; };           \
        _handle->dispatch_kern(_kern);           \
    } while (0)

//! disptch kern on current opr
#define MEGDNN_DISPATCH_CPU_KERN_OPR(_stmt) \
    MEGDNN_DISPATCH_CPU_KERN(               \
            static_cast<::megdnn::naive::HandleImpl*>(handle()), _stmt)

/*!
 * \brief operator impls should utilize this method to
 * \param _handle a pointer to HandleImpl
 * \param _parallelism the parallelism of task
 * \param _stmt the func to be executed for the kernel
 */
#define MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(_handle, _parallelism, _stmt) \
    do {                                                                    \
        _handle->dispatch_kern(_stmt, _parallelism);                        \
    } while (0)

//! disptch kern on current opr
#define MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN_OPR(_stmt, _parallelism)         \
    MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                     \
            static_cast<::megdnn::naive::HandleImpl*>(handle()), _parallelism, \
            _stmt)

// vim: syntax=cpp.doxygen
