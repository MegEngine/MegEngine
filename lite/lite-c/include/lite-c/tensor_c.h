/**
 * \file lite-c/include/lite-c/tensor_c.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#ifndef LITE_TENSOR_C_H_
#define LITE_TENSOR_C_H_

#include "common_enum_c.h"
#include "macro.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"
#include "stdint.h"

#define LAYOUT_MAX_DIM (7)

/*!
 * \brief the simple layout description
 */
typedef struct LiteLayout {
    size_t shapes[LAYOUT_MAX_DIM];
    size_t ndim;
    LiteDataType data_type;
} LiteLayout;

//! define a default LiteLayout
extern LITE_API const LiteLayout default_layout;

/*!
 * \brief warpper of the MegEngine Tensor
 *
 * if is_pinned_host is set, the storage memory of the tensor is pinned memory,
 * this is used to Optimize the H2D or D2H memory copy, if the device or layout
 * is not set, when copy form other device(CUDA, OpenCL) tensor, this tensor
 * will be automatically set to pinned tensor
 */
typedef struct LiteTensorDesc {
    //! flag whether the storage of the tensor is pinned, this is only used when
    //! the compnode is not in CPU
    int is_pinned_host;

    //! the layout of the tensor
    LiteLayout layout;

    //! the device of the tensor should not be changed after the tensor has
    //! constructed
    LiteDeviceType device_type;

    //! device id of the tensor
    int device_id;
} LiteTensorDesc;

//! define a default TensorDesc
extern LITE_API const LiteTensorDesc default_desc;

/*!
 * \brief The pointer to a Lite Tensor object
 */
typedef void* LiteTensor;

/**
 * \brief Create a lite tensor object from the given describe.
 * \param[in] tensor_describe The description to create the Tensor
 * \param[out] tensor The Tensor pointer
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_make_tensor(const LiteTensorDesc tensor_describe,
                              LiteTensor* tensor);

/**
 * \brief Destroy a lite tensor object.
 * \param[in] tensor The Tensor pointer
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_destroy_tensor(LiteTensor tensor);

/**
 * \brief change the layout of a Tensor object.
 * \param[in] tensor The Tensor
 * \param[out] layout The Layout to be set to a tensor
 */
LITE_API int LITE_set_tensor_layout(LiteTensor tensor, const LiteLayout layout);

/**
 * \brief use the user allocated data to reset the memory of the tensor, the
 * memory will not be managed by the lite, later, the user should delete
 * it.
 * \param[in] tensor The Tensor
 * \param[in] prepared_data The allocated memory which satisfy the Tensor
 * \param[in] data_length_in_byte The length of the allocated memory
 * layout
 */
LITE_API int LITE_reset_tensor_memory(LiteTensor tensor, void* prepared_data,
                                      size_t data_length_in_byte);

/**
 * \brief  use the user allocated data and corresponding layout to reset the
 * data and layout of the tensor, the memory will not be managed by lite, later,
 * the user should delete it.
 * \param[in] tensor The Tensor
 * \param[in] layout The Layout to be set to the tensor
 * \param[in] prepared_data The allocated memory which satisfy the layout to be
 * set
 */
LITE_API int LITE_reset_tensor(LiteTensor tensor, const LiteLayout layout,
                               void* prepared_data);

/**
 * \brief reshape a tensor with the memroy not change, the total number of
 * element in the reshaped tensor must equal to the origin tensor, the input
 * shape must only contain one or zero -1 to flag it can be deduced
 * automatically.
 * \param[in] tensor The Tensor to be reshape
 * \param[in] shape the user input shape
 * \param[in] size the number of data in shape,
 */
LITE_API int LITE_tensor_reshape(LiteTensor tensor, const int* shape, int size);

/**
 * \brief slice a tensor with input param
 * \param[in] tensor The Tensor to be slice
 * \param[in] start start index of every axis of to be sliced
 * \param[in] end end index of every axis of to be sliced
 * \param[in] step step of every axis of to be sliced, if nullptr, step will be
 * 1
 * \param[in] size the number axis to be sliced
 * \param[out] sliced_tensor the result tensor sliced from the origin tensor
 */
LITE_API int LITE_tensor_slice(const LiteTensor tensor, const size_t* start,
                               const size_t* end, const size_t* step,
                               size_t size, LiteTensor* slice_tensor);

/**
 * \brief fill zero to the tensor
 * \param[in] tensor The Tensor to be memset
 */
LITE_API int LITE_tensor_fill_zero(LiteTensor tensor);

/**
 * \brief copy tensor form other tensor
 * \param[out] dst_tensor The Tensor to copy into
 * \param[in] src_tensor The Tensor to copy from
 */
LITE_API int LITE_tensor_copy(LiteTensor dst_tensor,
                              const LiteTensor src_tensor);

/**
 * \brief share memory form other tensor
 * \param[out] dst_tensor The Tensor to share into
 * \param[in] src_tensor The Tensor to be shared
 */
LITE_API int LITE_tensor_share_memory_with(LiteTensor dst_tensor,
                                           const LiteTensor src_tensor);

/**
 * \brief get the memory pointer of a Tensor object.
 * \param[in] tensor The input Tensor
 * \param[out] data a pointer to void pointer
 */
LITE_API int LITE_get_tensor_memory(const LiteTensor tensor, void** data);

/**
 * \brief get the memory pointer of a Tensor object.
 * \param[in] tensor The input Tensor
 * \param[in] index The coordinate in the tensor
 * \param[in] size The lenght of coordinate
 * \param[out] data a pointer to void pointer
 */
LITE_API int LITE_get_tensor_memory_with_index(const LiteTensor tensor,
                                               const size_t* index, size_t size,
                                               void** data);

/**
 * \brief get the tensor capacity in byte of a Tensor object.
 * \param[in] tensor The input Tensor
 * \param[out] size_ptr a pointer to the return size

 */
LITE_API int LITE_get_tensor_total_size_in_byte(const LiteTensor tensor,
                                                size_t* size);

/**
 * \brief get the tensor layout of a Tensor object.
 * \param[in] tensor The input Tensor
 * \param[out] layout_ptr a pointer will be write with the layout of the tensor
 */
LITE_API int LITE_get_tensor_layout(const LiteTensor tensor,
                                    LiteLayout* layout);

/**
 * \brief get the tensor device of a Tensor object.
 * \param[in] tensor The input Tensor
 * \param[out] device_ptr a pointer will be write with the device of the tensor
 */
LITE_API int LITE_get_tensor_device_type(const LiteTensor tensor,
                                         LiteDeviceType* device_type);

/**
 * \brief get the tensor device id of a Tensor object.
 * \param[in] tensor The input Tensor
 * \param[out] device_id a pointer will be write with the device id of the
 * tensor
 */
LITE_API int LITE_get_tensor_device_id(const LiteTensor tensor, int* device_id);

/**
 * \brief whether the tensor is is_pinned_host.
 * \param[in] tensor The input Tensor
 * \param[out] is_pinned_host_ptr a int pointer will be write with whether the
 * tensor is pinned host
 */
LITE_API int LITE_is_pinned_host(const LiteTensor tensor, int* is_pinned_host);

/**
 * \brief whether the tensor memory is continue.
 * \param[in] tensor The input Tensor
 * \param[out] is_continue a int pointer will be write with whether the
 * tensor continue
 */
LITE_API int LITE_is_memory_continue(const LiteTensor tensor, int* is_continue);
/**
 * \brief concat the inputs tensor to one big tensor
 * \param[in] tensors ptr The input Tensors
 * \param[in] nr_tensors number input Tensor
 * \param[in] dim the dim concat act on
 * \param[in] dst_device the device type of result tensor, when
 * LITE_DEVICE_DEFAULT, the result tensor device type will get from the first
 * tensor
 * \param[in] device_id the device id of result tensor, when -1, the result
 * tensor device id will get from the first tensor
 * \param[out] result_tensor the result tensor after concat
 */
LITE_API int LITE_tensor_concat(LiteTensor* tensors, int nr_tensor, int dim,
                                LiteDeviceType dst_device, int device_id,
                                LiteTensor* result_tensor);

#ifdef __cplusplus
}
#endif
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
