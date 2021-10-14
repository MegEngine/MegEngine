/**
 * \file src/serialization/test/extern_c_opr_v23.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef MEGBRAIN_EXTERN_C_OPR_H
#define MEGBRAIN_EXTERN_C_OPR_H

#include <stddef.h>
#include <stdint.h>

#define MGB_PUBLIC __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

#define MGB_EXTERN_C_OPR_VERSION 0x23
#define MGB_TENSOR_MAX_NDIM      8

//! data types
typedef enum MGBDType { MGB_DTYPE_FLOAT32, MGB_DTYPE_INT32 } MGBDType;

typedef struct MGBTensorShape {
    uint32_t ndim, shape[MGB_TENSOR_MAX_NDIM];
} MGBTensorShape;

typedef struct MGBTensorLayout {
    uint32_t dtype;
    MGBTensorShape shape;
} MGBTensorLayout;

//! tensor representation
typedef struct MGBTensor {
    MGBTensorLayout layout;
    void* data;  //!< the tensor value, accessible by caller CPU thread
} MGBTensor;

typedef struct ExternCOprParam {
    //! just for build
    size_t _;
} ExternCOprParam;
/*!
 * \brief operator descriptor
 *
 * Note: all the methods (except release) should be purely functional, so a
 * descriptor can be shared by multiple operators
 */
typedef struct MGBOprDesc {
    //! number of input/output vars
    size_t nr_input, nr_output;

    //! operator type name
    const char* type_name;

    //! release this descriptor
    void (*release)(struct MGBOprDesc* self);

    //! compute hash
    size_t (*hash)(const struct MGBOprDesc* self);

    //! equality check
    int (*is_same)(const struct MGBOprDesc* self, const struct MGBOprDesc* rhs);

    //! perform the computation
    void (*execute)(
            const struct MGBOprDesc* self, const MGBTensor* input, MGBTensor* output);

    //! infer output shapes from input shapes
    void (*infer_shape)(
            const struct MGBOprDesc* self, const MGBTensorShape* input,
            MGBTensorShape* output);

    //! custom user data to be associated with this descriptor
    void* user_data;

    //! dynamic extern c opr param
    ExternCOprParam* dynamic_param;
} MGBOprDesc;

//! foreach member function of MGBOprDesc to help initialization
#define MGB_OPR_DESC_FOREACH_MEM_FN(cb) \
    cb(release) cb(hash) cb(is_same) cb(execute) cb(infer_shape)

//! operator loader
typedef struct MGBOprLoader {
    //! name of the loader; must match the name given in
    //! ExternCOprRunner::make_placeholder and would be written to graph dump
    //! file
    const char* name;

    //! create a new descriptor from saved buffer
    MGBOprDesc* (*create_desc)(size_t nr_input, const void* buf, size_t buf_len);
} MGBOprLoader;

//! APIs provided by megbrain
typedef struct MGBExternCOprApi {
    /*!
     * \brief register an operator loader
     *
     * content of the loader would be copied
     *
     * \return true if registration succeeds; false if duplicated name
     */
    int (*register_loader)(const MGBOprLoader* loader);

    /*!
     * \brief unregister a MGBOprLoader
     * \return whether any loader is removed (i.e. whether the name exists)
     */
    int (*unregister_loader)(const char* name);
} MGBExternCOprApi;

//! get API ptr for specific version; return nullptr if version mismatch
MGB_PUBLIC const MGBExternCOprApi* mgb_get_extern_c_opr_api_versioned(int version);

#ifdef __cplusplus
}
#endif

//! get the API ptr for current header version; return nullptr on mismatch
static inline const MGBExternCOprApi* mgb_get_extern_c_opr_api() {
    return mgb_get_extern_c_opr_api_versioned(MGB_EXTERN_C_OPR_VERSION);
}

static inline size_t mgb_get_dtype_size(MGBDType dtype) {
    switch (dtype) {
        case MGB_DTYPE_INT32:
            return 4;
        case MGB_DTYPE_FLOAT32:
            return 4;
        default:
            __builtin_trap();
            return -1;
    }
}

#undef MGB_PUBLIC
#endif  // MEGBRAIN_EXTERN_C_OPR_H

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
