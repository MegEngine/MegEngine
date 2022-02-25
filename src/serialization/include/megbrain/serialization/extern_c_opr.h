#ifndef MEGBRAIN_EXTERN_C_OPR_H
#define MEGBRAIN_EXTERN_C_OPR_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef MGE_DLL_EXPORT
#define MGB_PUBLIC __declspec(dllexport)
#else
#define MGB_PUBLIC __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MGB_C_OPR_INIT_FUNC
#define MGB_C_OPR_INIT_FUNC mgb_c_opr_init
#endif

#define INIT_FUNCS(s)           #s
#define INIT_FUNC(s)            INIT_FUNCS(s)
#define MGB_C_OPR_INIT_FUNC_STR INIT_FUNC(MGB_C_OPR_INIT_FUNC)

#define MGB_EXTERN_C_OPR_VERSION 0x24
#define MGB_TENSOR_MAX_NDIM      8

//! data types
typedef enum MGBDType {
    MGB_DTYPE_FLOAT32,
    MGB_DTYPE_INT32,
    MGB_DTYPE_UINT8,
    //! IEEE 754-based half-precision floating
    MGB_DTYPE_FLOAT16,
    MGB_DTYPE_INT16,
} MGBDType;

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

//! extern device tenosr struct
typedef struct ExternDeviceTensor {
    //! layout of device extern tensor, use to validity check with MGBTensor
    MGBTensorLayout layout;
    //! different NPU API has different type define so just define a void * to
    //! compat all, need loader and SDK implement reinterpret_cast it
    //! exampe for NNIE, device_ptr may define as
    //! struct MemoryInfo {
    //!     HI_U64 phy_addr;
    //!     void* vir_addr;
    //!     size_t size = 0;
    //! }
    void* device_ptr;
} ExternDeviceTensor;

//! for dynamic extern c opr param
typedef struct ExternCOprParam {
    //! dump name of extern c opr in graph
    //! example graph:
    //! ExternCOpr1(3516:preprocess)->opr->ExternCOpr2(3559)->opr->ExternCOpr3(3516:det_face)...
    //! extern_c_opr_dump_name config case:
    //! when set 3516:preprocess, ExternCOpr1 will be config.
    //! when set 3559, ExternCOpr2 will be config.
    //! when set 3516:det_face, ExternCOpr3 will be config.
    //! when set nullptr, will auto config the first ExternCOpr.
    const char* extern_c_opr_dump_name;

    //! number of input/output, use to index and check
    //! if set nr_input = 0, means do not provide input ExternDeviceTensor
    //! if set nr_output = 0, means do not provide nr_output ExternDeviceTensor
    size_t nr_input, nr_output;

    //! ptr of input/output ExternDeviceTensor
    ExternDeviceTensor* input;
    ExternDeviceTensor* output;

    //! device id
    size_t device_id;

    //! extra info for misc dynamic config
    uint8_t* extra_info;
    //! size of extra_info
    size_t extra_info_size;
} ExternCOprParam;

/*!
 * \brief operator descriptor
 *
 * Note: all the methods (except release) should be purely functional, so a
 * descriptor can be shared by multiple operators
 */
typedef struct MGBOprDesc {
    //! size of this MGBOprDesc object
    uint32_t size;

    //! number of input/output vars
    uint32_t nr_output;

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
            const struct MGBOprDesc* self, const MGBTensor* input,
            const MGBTensor* output);

    //! infer output shapes from input shapes
    void (*infer_shape)(
            const struct MGBOprDesc* self, const MGBTensorShape* input,
            MGBTensorShape* output);

    //! optional: infer output dtypes from input dtypes
    void (*infer_dtype)(
            const struct MGBOprDesc* self, const MGBDType* input, MGBDType* output);

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

    /*!
     * \brief create a new descriptor from saved buffer
     *
     * Note: there is no guarantee on the alignment of \p buf.
     */
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
        case MGB_DTYPE_UINT8:
            return 1;
        case MGB_DTYPE_FLOAT16:
        case MGB_DTYPE_INT16:
            return 2;
        default:
            __builtin_trap();
            return -1;
    }
}

static inline void mgb_init_opr_desc(
        MGBOprDesc* desc, uint32_t nr_output, const char* type_name) {
    memset(desc, 0, sizeof(MGBOprDesc));
    desc->size = sizeof(MGBOprDesc);
    desc->nr_output = nr_output;
    desc->type_name = type_name;
}

#undef MGB_PUBLIC
#endif  // MEGBRAIN_EXTERN_C_OPR_H

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
