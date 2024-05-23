#pragma once

#include <type_traits>

#include "acl/acl.h"
#include "acl/acl_op.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#include "megdnn/basic_types.h"
#include "megdnn/common.h"
#include "src/atlas/handle.h"
#include "src/atlas/utils.h"
#include "src/common/metahelper.h"
#include "src/common/utils.cuh"

namespace megdnn {
namespace atlas {

// clang-format off
#define FOR_ACL_C_DTYPE_MAP(cb) \
    cb(ACL_INT64, int64_t)      \
    cb(ACL_INT32, int32_t)      \
    cb(ACL_INT16, int16_t)      \
    cb(ACL_INT8, int8_t)        \
    cb(ACL_UINT64, uint64_t)    \
    cb(ACL_UINT32, uint32_t)    \
    cb(ACL_UINT16, uint16_t)    \
    cb(ACL_UINT8, uint8_t)      \
    cb(ACL_DOUBLE, double)      \
    cb(ACL_FLOAT, float)        \
    cb(ACL_BOOL, bool)

#define FOR_ACL_DNN_DTYPE_MAP(cb)             \
    cb(ACL_INT32, Int32)                      \
    cb(ACL_INT16, Int16)                      \
    cb(ACL_INT8, Int8)                        \
    cb(ACL_UINT16, Uint16)                    \
    cb(ACL_UINT8, Uint8)                      \
    DNN_INC_FLOAT16(cb(ACL_FLOAT16, Float16)) \
    DNN_INC_FLOAT16(cb(ACL_BF16, BFloat16))   \
    cb(ACL_FLOAT, Float32)                    \
    cb(ACL_BOOL, Bool)
// clang-format on

template <typename T>
struct c2acl_type;
template <aclDataType>
struct acl2c_type;
template <DTypeEnum>
struct dnn2acl_type;
template <aclDataType>
struct acl2dnn_type;

#define IMPL(acltype, ctype)                         \
    template <>                                      \
    struct c2acl_type<ctype> {                       \
        static constexpr aclDataType type = acltype; \
    };                                               \
    template <>                                      \
    struct acl2c_type<acltype> {                     \
        using type = ctype;                          \
    };

FOR_ACL_C_DTYPE_MAP(IMPL);
DNN_INC_FLOAT16(IMPL(ACL_FLOAT16, dt_float16))

#undef IMPL

#define IMPL(acltype, dnntype)                                \
    template <>                                               \
    struct dnn2acl_type<DTypeEnum::dnntype> {                 \
        static constexpr aclDataType type = acltype;          \
    };                                                        \
    template <>                                               \
    struct acl2dnn_type<acltype> {                            \
        static constexpr DTypeEnum type = DTypeEnum::dnntype; \
    };

FOR_ACL_DNN_DTYPE_MAP(IMPL);

#undef IMPL

template <typename T>
using c2acl_t = typename c2acl_type<T>::type;
template <aclDataType dt>
using acl2c_t = typename acl2c_type<dt>::type;
template <DTypeEnum dt>
using dnn2acl_t = typename dnn2acl_type<dt>::type;
template <aclDataType dt>
using acl2dnn_t = typename acl2dnn_type<dt>::type;

aclDataType as_acl_dtype(DTypeEnum dtype);
aclDataType as_acl_dtype(DType dtype);
DType as_dnn_dtype(aclDataType dtype);

#define ACL_RAII_DECLARE(wrapper_cls, acl_t, create, destroy, checker) \
    class wrapper_cls : public NonCopyableObj {                        \
        acl_t* m_impl = nullptr;                                       \
                                                                       \
    public:                                                            \
        wrapper_cls(wrapper_cls&& other) {                             \
            if (m_impl) {                                              \
                checker(destroy(m_impl));                              \
            }                                                          \
            m_impl = other.m_impl;                                     \
            other.m_impl = nullptr;                                    \
        }                                                              \
        wrapper_cls& operator=(wrapper_cls&& other) {                  \
            if (m_impl) {                                              \
                checker(destroy(m_impl));                              \
            }                                                          \
            m_impl = other.m_impl;                                     \
            other.m_impl = nullptr;                                    \
            return *this;                                              \
        }                                                              \
        acl_t* get() const { return m_impl; }                          \
        ~wrapper_cls() {                                               \
            if (m_impl) {                                              \
                checker(destroy(m_impl));                              \
            }                                                          \
            m_impl = nullptr;                                          \
        }

// clang-format off
ACL_RAII_DECLARE(AclTensor, aclTensor, aclCreateTensor, aclDestroyTensor, aclnn_check) // {
    AclTensor() = default;

    AclTensor(void* devptr, const TensorLayout& layout, 
            aclFormat acl_format = aclFormat::ACL_FORMAT_ND,
            aclDataType acl_type = ACL_DT_UNDEFINED) {
        init(devptr, layout, acl_format, acl_type);
    }

    AclTensor(const TensorND& src, aclFormat acl_format = aclFormat::ACL_FORMAT_ND, aclDataType acl_type = ACL_DT_UNDEFINED)  {
        init(src, acl_format, acl_type);
    }

    void init(void* devptr, const TensorLayout& layout, aclFormat acl_format = aclFormat::ACL_FORMAT_ND,
            aclDataType acl_type = ACL_DT_UNDEFINED) {
        megdnn_assert(m_impl == nullptr, "AclTensor has already been initialized");

        megdnn::SmallVector<int64_t> shape(layout.ndim), stride(layout.ndim);
        for (size_t i = 0; i < layout.ndim; ++i) {
            shape[i] = static_cast<int64_t>(layout[i]);
            stride[i] = static_cast<int64_t>(layout.stride[i]);
        }
        auto layout_span = layout.span();
        int64_t offset = -1 * layout_span.low_elem;
        if (layout.is_contiguous()) {
            //! for resize opr
            m_impl = aclCreateTensor(
                    shape.data(), layout.ndim, acl_type == ACL_DT_UNDEFINED ? as_acl_dtype(layout.dtype):acl_type, stride.data(),
                    offset, acl_format, shape.data(), shape.size(), 
                    static_cast<void*>(static_cast<uint8_t*>(devptr) - layout.dtype.size() * offset)
            );
        } else {
            megdnn::SmallVector<int64_t> storage_shape(1, layout_span.dist_elem());
            m_impl = aclCreateTensor(
                    shape.data(), layout.ndim, acl_type == ACL_DT_UNDEFINED ? as_acl_dtype(layout.dtype):acl_type, stride.data(),
                    offset, acl_format, storage_shape.data(), storage_shape.size(), 
                    static_cast<void*>(static_cast<uint8_t*>(devptr) - layout.dtype.size() * offset)
            );
        }
        megdnn_assert(m_impl, "construct aclTensor failed");
    }

    void init(const TensorND& src, aclFormat acl_format = aclFormat::ACL_FORMAT_ND, aclDataType acl_type = ACL_DT_UNDEFINED) {
        init(src.raw_ptr(), src.layout, acl_format, acl_type);
    }

    std::string to_string() const {
        megdnn_assert(m_impl, "construct aclTensor failed");
        int64_t* shape_value = nullptr;
        uint64_t shape_ndim = 0;
        acl_check(aclGetViewShape(m_impl, &shape_value, &shape_ndim));
        int64_t* stride_value = nullptr;
        uint64_t stride_ndim = 0;
        acl_check(aclGetViewStrides(m_impl, &stride_value, &stride_ndim));
        int64_t* storage_value = nullptr;
        uint64_t storage_ndim = 0;
        acl_check(aclGetStorageShape(m_impl, &storage_value, &storage_ndim));
        int64_t offset = 0;
        acl_check(aclGetViewOffset(m_impl, &offset));
        aclDataType data_type = ACL_DT_UNDEFINED;
        acl_check(aclGetDataType(m_impl, &data_type));
        aclFormat fmt = ACL_FORMAT_UNDEFINED;
        acl_check(aclGetFormat(m_impl, &fmt));

        auto str_list = [](const int64_t* values, uint64_t ndim) -> std::string {
            std::string str = "{";
            for (size_t i = 0; i < ndim; ++i) {
                str += std::to_string(values[i]);
                if (i + 1 != ndim) {
                    str += ", ";
                }
            }
            str += "}";
            return str;
        };
        
        std::string shape_str = str_list(shape_value, shape_ndim);
        std::string stride_str = str_list(stride_value, stride_ndim);
        std::string storage_str = str_list(storage_value, storage_ndim);

        char ret[128];
        sprintf(ret, "shape: %s, stride: %s, storage: %s, offset: %ld, dtype: %s", 
                shape_str.c_str(), stride_str.c_str(), storage_str.c_str(), offset,
                as_dnn_dtype(data_type).name());
        return std::string(ret);
    }
};

ACL_RAII_DECLARE(AclTensorDesc, aclTensorDesc, aclCreateTensorDesc, aclDestroyTensorDesc, void) // {
    AclTensorDesc(const TensorLayout &layout) {
        SmallVector<int64_t> shape(layout.ndim);
        for (size_t i = 0; i < layout.ndim; ++i) {
            shape[i] = static_cast<int64_t>(layout[i]);
        }
        m_impl = aclCreateTensorDesc(as_acl_dtype(layout.dtype), layout.ndim,
                shape.data(), aclFormat::ACL_FORMAT_ND);
        megdnn_assert(m_impl, "construct aclTensorDesc failed"); 
    }
};

ACL_RAII_DECLARE(AclDataBuffer, aclDataBuffer, aclCreateDataBuffer, aclDestroyDataBuffer, acl_check) // {
    AclDataBuffer(const TensorND &data) {
        auto &&layout = data.layout;
        m_impl = aclCreateDataBuffer(data.raw_ptr(), 
                layout.total_nr_elems() * layout.dtype.size());
        megdnn_assert(m_impl, "construct aclDataBuffer failed");
    }
};

ACL_RAII_DECLARE(AclopAttr, aclopAttr, aclopCreateAttr, aclopDestroyAttr, void) // {
    AclopAttr() {
        m_impl = aclopCreateAttr();
        megdnn_assert(m_impl, "construct aclopAttr failed");
    }
    AclopAttr &set(const std::string &name, bool value) {
        acl_check(aclopSetAttrBool(m_impl, name.c_str(), value));
        return *this;
    }
    AclopAttr &set(const std::string &name, int64_t value) {
        acl_check(aclopSetAttrInt(m_impl, name.c_str(), value));
        return *this;
    }
    AclopAttr &set(const std::string &name, float value) {
        acl_check(aclopSetAttrFloat(m_impl, name.c_str(), value));
        return *this;
    }
    AclopAttr &set(const std::string &name, const std::string &value) {
        acl_check(aclopSetAttrString(m_impl, name.c_str(), value.c_str()));
        return *this;
    }
};

ACL_RAII_DECLARE(AclScalar, aclScalar, aclCreateScalar, aclDestroyScalar, aclnn_check) // {
    template <typename T>
    AclScalar(T val) {
        m_impl = aclCreateScalar(&val, c2acl_type<std::decay_t<T>>::type);
        megdnn_assert(m_impl, "construct AclScalar failed");
    }
    template <typename T>
    AclScalar(T val, aclDataType acl_dtype) {
#define cb(acltype, ctype)                                   \
    case acltype: {                                          \
        auto new_val = static_cast<ctype>(val);              \
        m_impl = aclCreateScalar(&new_val, acl_dtype);       \
        megdnn_assert(m_impl, "construct AclScalar failed"); \
        break;                                               \
    }
        switch (acl_dtype) {
            FOR_ACL_C_DTYPE_MAP(cb)
            DNN_INC_FLOAT16(cb(ACL_FLOAT16, dt_float16))
            default:
                megdnn_throw("unsupported dtype");
        }
#undef cb
    }
    template <typename T>
    AclScalar(T val, DType dnn_dt) : AclScalar(val, as_acl_dtype(dnn_dt)) {}
};

ACL_RAII_DECLARE(
        AclIntArray, aclIntArray, aclCreateIntArray, aclDestroyIntArray, aclnn_check) // {
    template <typename T>
    AclIntArray(const SmallVector<T> &value) {
        static_assert(std::is_integral_v<T> == true, "construct AclIntArray with non-int input");
        m_vector.reserve(value.size());
        for (auto element: value) {
            m_vector.push_back(static_cast<int64_t>(element));
        }
        m_impl = aclCreateIntArray(m_vector.data(), m_vector.size());
        megdnn_assert(m_impl, "construct AclIntArray failed");
    }

    template <typename T>
    AclIntArray(std::initializer_list<T> value) {
        static_assert(std::is_integral_v<T> == true, "construct AclIntArray with non-int input");
        m_vector.reserve(value.size());
        for (auto element: value) {
            m_vector.push_back(static_cast<int64_t>(element));
        }
        m_impl = aclCreateIntArray(m_vector.data(), m_vector.size());
        megdnn_assert(m_impl, "construct AclIntArray failed");
    }

    template <typename T>
    AclIntArray(const T* src, size_t sz) {
        static_assert(std::is_integral_v<T> == true, "construct AclIntArray with non-int input");
        m_vector.reserve(sz);
        for (size_t i=0; i<sz; ++i) {
            m_vector.push_back(static_cast<int64_t>(src[i]));
        }
        m_impl = aclCreateIntArray(m_vector.data(), m_vector.size());
        megdnn_assert(m_impl, "construct AclIntArray failed");
    }
private:
    SmallVector<int64_t> m_vector;
};

ACL_RAII_DECLARE(
        AclFloatArray, aclFloatArray, aclCreateFloatArray, aclDestroyFloatArray, aclnn_check) // {
    template <typename T>
    AclFloatArray(const SmallVector<T> &value) {
        static_assert(std::is_floating_point_v<T> == true, "construct AclFloatArray with non-float input");
        m_vector.reserve(value.size());
        for (auto element: value) {
            m_vector.push_back(static_cast<float>(element));
        }
        m_impl = aclCreateFloatArray(m_vector.data(), m_vector.size());
        megdnn_assert(m_impl, "construct AclFloatArray failed");
    }
    template <typename T>
    AclFloatArray(const T* src, size_t sz) {
        static_assert(std::is_floating_point_v<T> == true, "construct AclFloatArray with non-float input");
        m_vector.reserve(sz);
        for (size_t i=0; i<sz; ++i) {
            m_vector.push_back(static_cast<float>(src[i]));
        }
        m_impl = aclCreateFloatArray(m_vector.data(), m_vector.size());
        megdnn_assert(m_impl, "construct AclFloatArray failed");
    }
private:
    SmallVector<float> m_vector;
};

ACL_RAII_DECLARE(
        AclBoolArray, aclBoolArray, aclCreateBoolArray, aclDestroyBoolArray, aclnn_check) // {
    AclBoolArray(const bool* src, size_t sz) {
        m_impl = aclCreateBoolArray(src, sz);
        megdnn_assert(m_impl, "construct AclBoolArray failed");
    }
    AclBoolArray(std::initializer_list<bool> values) {
        static_assert(sizeof(uint8_t) == sizeof(bool));
        for (auto val: values) {
            m_vector.push_back(static_cast<uint8_t>(val));
        }
        m_impl = aclCreateBoolArray(reinterpret_cast<bool*>(m_vector.data()), m_vector.size());
        megdnn_assert(m_impl, "construct AclBoolArray failed");
    }
    AclBoolArray(const SmallVector<uint8_t> &values) {
        static_assert(sizeof(uint8_t) == sizeof(bool));
        for (auto val: values) {
            m_vector.push_back(val);
        }
        m_impl = aclCreateBoolArray(reinterpret_cast<bool*>(m_vector.data()), m_vector.size());
        megdnn_assert(m_impl, "construct AclBoolArray failed");
    }
private:
    SmallVector<uint8_t> m_vector;
};

ACL_RAII_DECLARE(
        AclTensorList, aclTensorList, aclCreateTensorList, aclDestroyTensorList, aclnn_check) // {
    AclTensorList(const std::vector<TensorND>& src, aclFormat acl_format = aclFormat::ACL_FORMAT_ND) {
        std::vector<aclTensor*> acl_src;
        for (size_t i = 0; i < src.size(); ++i) {
            auto devptr = src[i].raw_ptr();
            auto layout = src[i].layout;
            megdnn::SmallVector<int64_t> shape(layout.ndim), stride(layout.ndim);
            for (size_t i = 0; i < layout.ndim; ++i) {
                shape[i] = static_cast<int64_t>(layout[i]);
                stride[i] = static_cast<int64_t>(layout.stride[i]);
            }
            auto layout_span = layout.span();
            megdnn::SmallVector<int64_t> storage_shape(1, layout_span.dist_elem());
            int64_t offset = -1 * layout_span.low_elem;
            auto acl_tensor = aclCreateTensor(
                    shape.data(), layout.ndim, as_acl_dtype(layout.dtype), stride.data(),
                    offset, acl_format, storage_shape.data(), 1, 
                    static_cast<void*>(static_cast<uint8_t*>(devptr) - layout.dtype.size() * offset)
            );
            acl_src.push_back(acl_tensor);
        }
        m_impl = aclCreateTensorList(acl_src.data(), acl_src.size());
        megdnn_assert(m_impl, "construct AclTensorList failed");
    }
};

ACL_RAII_DECLARE(
        AclScalarList, aclScalarList, aclCreateScalarList, aclDestroyScalarList, aclnn_check) // {
    // to be implemented
};

// clang-format on

#undef ACL_RAII_DECLARE

class AclMem : public NonCopyableObj {
    void* m_ptr = nullptr;
    atlas::HandleImpl* m_handle = nullptr;
    size_t m_size = 0;

public:
    AclMem() = default;
    AclMem(size_t size_in_bytes, atlas::HandleImpl* handle,
           aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST) {
        alloc(size_in_bytes, handle, policy);
    }
    ~AclMem() { free(); }

    void alloc(
            size_t size_in_bytes, atlas::HandleImpl* handle,
            aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST) {
        megdnn_assert(m_ptr == nullptr, "workspace is already allocted");
        m_handle = handle;
        m_size = size_in_bytes;
        m_ptr = m_handle->alloc(size_in_bytes, policy);
    }

    void free() {
        if (m_ptr) {
            m_handle->free(m_ptr);
        }
        m_ptr = nullptr;
        m_handle = nullptr;
    }

    void* ptr() const { return m_ptr; }
};

class AclOprRunner {
    atlas::HandleImpl* m_handle;
    std::string m_op_type;
    AclopAttr m_op_attr;
    megdnn::SmallVector<AclTensorDesc> m_input_desc;
    megdnn::SmallVector<AclDataBuffer> m_input_buf;
    megdnn::SmallVector<AclTensorDesc> m_output_desc;
    megdnn::SmallVector<AclDataBuffer> m_output_buf;

public:
    AclOprRunner(const std::string& op_type, atlas::HandleImpl* handle)
            : m_handle(handle), m_op_type(op_type) {}
    AclOprRunner& add_input(const TensorND& input) {
        m_input_desc.emplace_back(input.layout);
        m_input_buf.emplace_back(input);
        return *this;
    }
    AclOprRunner& add_output(const TensorND& output) {
        m_output_desc.emplace_back(output.layout);
        m_output_buf.emplace_back(output);
        return *this;
    }
    void run() {
        auto stream = m_handle->stream();
        SmallVector<aclTensorDesc*> inp_desc, oup_desc;
        SmallVector<aclDataBuffer*> inp_buf, oup_buf;

        for (size_t i = 0; i < m_input_desc.size(); ++i) {
            inp_desc.emplace_back(m_input_desc[i].get());
            inp_buf.emplace_back(m_input_buf[i].get());
        }
        for (size_t i = 0; i < m_output_desc.size(); ++i) {
            oup_desc.emplace_back(m_output_desc[i].get());
            oup_buf.emplace_back(m_output_buf[i].get());
        }

        acl_check(aclopCompileAndExecute(
                m_op_type.c_str(), inp_desc.size(), inp_desc.data(), inp_buf.data(),
                oup_desc.size(), oup_desc.data(), oup_buf.data(), m_op_attr.get(),
                ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream));
    }
};

#define CUBE_KEEP_DTYPE                0
#define CUBE_ALLOW_FP32_DOWN_PRECISION 1
#define CUBE_USE_FP16                  2
#define CUBE_USE_FP32                  3

}  // namespace atlas
}  // namespace megdnn
