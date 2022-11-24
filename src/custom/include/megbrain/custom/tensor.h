#pragma once

#include <string>
#include <vector>
#include "accessor.h"
#include "utils.h"

namespace custom {

#define CUSTOM_DATA_ADAPTOR_FRIEND_DECL                \
    template <typename BuiltinT, typename CustomT>     \
    friend BuiltinT to_builtin(const CustomT& custom); \
    template <typename BuiltinT, typename CustomT>     \
    friend CustomT to_custom(const BuiltinT& builtin)

#define CUSTOM_FOR_EACH_DEVICE_TYPE(cb) cb(x86, CPU, "cpux") cb(cuda, CUDA, "gpux")

#define CUSTOM_DEVICE_TYPE_ENUM_DECL(custom_type, builtin_type, builtin_str) \
    custom_type,

class MGE_WIN_DECLSPEC_FUC Device {
    const void* impl() const;
    Device(const void* impl);
    CUSTOM_PIMPL_CLS_DECL(Device);

public:
    enum class DeviceEnum : uint32_t {
        CUSTOM_FOR_EACH_DEVICE_TYPE(CUSTOM_DEVICE_TYPE_ENUM_DECL)
    };

    Device(const std::string& device);
    Device(const char* device);
    Device(DeviceEnum device);

    std::string str(void) const;
    DeviceEnum enumv(void) const;

    static bool is_legal(const std::string& device);
    static bool is_legal(DeviceEnum device);
    static std::vector<std::string> legal_devices(void);

    friend class Tensor;
    MGE_WIN_DECLSPEC_FUC friend bool operator==(const Device& lhs, const Device& rhs);
    CUSTOM_DATA_ADAPTOR_FRIEND_DECL;
};

using DeviceEnum = Device::DeviceEnum;

bool operator==(const Device& lhs, const Device& rhs);

class MGE_WIN_DECLSPEC_FUC Shape {
    const void* impl() const;
    Shape(const void* impl);
    CUSTOM_PIMPL_CLS_DECL(Shape);

public:
    Shape(const std::vector<size_t>& rhs);
    Shape(const std::initializer_list<size_t>& rhs);

    size_t& operator[](size_t idx);
    size_t operator[](size_t idx) const;

    void ndim(size_t dim);
    size_t ndim(void) const;

    friend class Tensor;
    MGE_WIN_DECLSPEC_FUC friend bool operator==(const Shape& lhs, const Shape& rhs);
    CUSTOM_DATA_ADAPTOR_FRIEND_DECL;
};

bool operator==(const Shape& lhs, const Shape& rhs);

using float16_t = uint16_t;
using bfloat16_t = uint16_t;

#if MEGDNN_DISABLE_FLOAT16
#define fp16_wrap(cb, custom_dtype, dnn_dtype, c_dtype)
#else
#define fp16_wrap(cb, custom_dtype, dnn_dtype, c_dtype) \
    cb(custom_dtype, dnn_dtype, c_dtype)
#endif

// clang-format off
#define CUSTOM_FOR_EACH_TENSOR_DATA_TYPE(cb)        \
    cb(float32, Float32, float)                     \
    cb(uint8, Uint8, uint8_t)                       \
    cb(int8, Int8, int8_t)                          \
    cb(int16, Int16, int16_t)                       \
    cb(int32, Int32, int32_t)                       \
    fp16_wrap(cb, float16, Float16, float16_t)      \
    fp16_wrap(cb, bfloat16, BFloat16, bfloat16_t)   \
    cb(uint16, Uint16, uint16_t)                    \
    cb(quint8, Quantized8Asymm, uint8_t)            \
    cb(qint32, QuantizedS32, int32_t)               \
    cb(qint8, QuantizedS8, int8_t)                  \
    cb(qint16, QuantizedS16, int16_t)
// clang-format on

#define CUSTOM_DTYPE_ENUM_DECL(custom_type, builtin_type, ctype) custom_type,

class MGE_WIN_DECLSPEC_FUC DType {
    const void* impl() const;
    DType(const void* impl);
    CUSTOM_PIMPL_CLS_DECL(DType);

public:
    enum class DTypeEnum : uint32_t {
        CUSTOM_FOR_EACH_TENSOR_DATA_TYPE(CUSTOM_DTYPE_ENUM_DECL)
    };

    DType(const std::string& dtype);
    DType(const char* dtype);
    DType(const std::string& dtype, float scale, uint8_t zero_point = 0);
    DType(const char* dtype, float scale, uint8_t zero_point = 0);
    DType(DTypeEnum dtype);
    DType(DTypeEnum dtype, float scale, uint8_t zero_point = 0);

    std::string str(void) const;
    DTypeEnum enumv() const;
    float scale(void) const;
    uint8_t zero_point(void) const;
    template <typename T>
    bool is_compatible(void) const;

    static bool is_legal(const std::string& dtype);
    static bool is_legal(const DTypeEnum& dtype);
    static std::vector<std::string> legal_dtypes(void);

    friend class Tensor;
    MGE_WIN_DECLSPEC_FUC friend bool operator==(const DType& lhs, const DType& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator==(
            const DType& lhs, const std::string& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator==(const DType& lhs, const char* rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator==(
            const std::string& lhs, const DType& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator==(const char* lhs, const DType& rhs);

    CUSTOM_DATA_ADAPTOR_FRIEND_DECL;
};

using DTypeEnum = DType::DTypeEnum;

template <DTypeEnum>
struct DTypeTrait;

#define CUSTOM_DEFINE_DTYPE_TRAIT(custom_type, builtin_type, ctype) \
    template <>                                                     \
    struct DTypeTrait<DTypeEnum::custom_type> {                     \
        using type = ctype;                                         \
    };

#define CUSTOM_CASE_TO_COMPARE_DTYPE(custom_type, builtin_type, ctype) \
    case (DTypeEnum::custom_type): {                                   \
        return std::is_same<DecayT, ctype>::value;                     \
    }

CUSTOM_FOR_EACH_TENSOR_DATA_TYPE(CUSTOM_DEFINE_DTYPE_TRAIT)

template <typename T>
bool DType::is_compatible(void) const {
    using DecayT = typename std::decay<T>::type;
    auto dtype_enum = enumv();
#if !MEGDNN_DISABLE_FLOAT16
    if (dtype_enum == DTypeEnum::float16) {
        return sizeof(DecayT) == sizeof(DTypeTrait<DTypeEnum::float16>::type);
    } else if (dtype_enum == DTypeEnum::bfloat16) {
        return sizeof(DecayT) == sizeof(DTypeTrait<DTypeEnum::bfloat16>::type);
    }
#endif
    switch (dtype_enum) {
        CUSTOM_FOR_EACH_TENSOR_DATA_TYPE(CUSTOM_CASE_TO_COMPARE_DTYPE)
        default:
            return false;
    }
}

bool operator==(const DType& lhs, const DType& rhs);
bool operator==(const DType& lhs, const std::string& rhs);
bool operator==(const DType& lhs, const char* rhs);
bool operator==(const std::string& lhs, const DType& rhs);
bool operator==(const char* lhs, const DType& rhs);

class MGE_WIN_DECLSPEC_FUC Format {
    const void* impl() const;
    Format(const void* impl);
    CUSTOM_PIMPL_CLS_DECL(Format);

public:
    Format(const std::string& format);
    Format(const char* format);

    std::string str(void) const;
    bool is_default(void) const;

    friend class Tensor;
    CUSTOM_DATA_ADAPTOR_FRIEND_DECL;
};

class MGE_WIN_DECLSPEC_FUC Tensor {
    void* m_tensor;

    const void* impl(void) const;
    Tensor(const void* impl);

    const size_t* shapes_raw(void) const;
    const ptrdiff_t* strides_raw(void) const;

public:
    Tensor() = delete;
    Tensor(const Tensor& rhs);
    Tensor& operator=(const Tensor& rhs);

    Shape shape(void) const;
    DType dtype(void) const;
    Format format(void) const;
    Device device(void) const;

    size_t size(void) const;
    std::vector<ptrdiff_t> stride(void) const;
    float scale(void) const;
    uint8_t zero_point(void) const;

    bool is_contiguous() const;
    bool is_empty() const;

    void* data(void) const;
    template <typename T>
    T* data(void) const;

    template <
            typename T, size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits,
            typename index_t = int64_t>
    const TensorAccessor<T, N, PtrTraits, index_t> accessor() const;

    template <
            typename T, size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits,
            typename index_t = int64_t>
    TensorAccessor<T, N, PtrTraits, index_t> accessor();

    CUSTOM_DATA_ADAPTOR_FRIEND_DECL;
};

template <typename T>
T* Tensor::data(void) const {
    custom_assert(
            dtype().is_compatible<T>(), "invalid convert, tensor data type is %s",
            dtype().str().c_str());
    return reinterpret_cast<T*>(data());
}

template <typename T, size_t N, template <typename U> class PtrTraits, typename index_t>
const TensorAccessor<T, N, PtrTraits, index_t> Tensor::accessor() const {
    return const_cast<Tensor*>(this)->accessor<T, N, PtrTraits, index_t>();
}

template <typename T, size_t N, template <typename U> class PtrTraits, typename index_t>
TensorAccessor<T, N, PtrTraits, index_t> Tensor::accessor() {
    custom_assert(
            N == shape().ndim(),
            "cannot get a %lu-d accessor for a tensor with dim %lu",
            static_cast<unsigned long>(N), static_cast<unsigned long>(shape().ndim()));
    custom_assert(N > 0, "cannot get 0-d accessor");

    T* ptr = data<T>();
    return TensorAccessor<T, N, PtrTraits, index_t>(ptr, shapes_raw(), strides_raw());
}

#undef CUSTOM_DATA_ADAPTOR_FRIEND_DECL
#undef CUSTOM_DEVICE_TYPE_ENUM_DECL
#undef CUSTOM_DTYPE_ENUM_DECL
#undef CUSTOM_DEFINE_DTYPE_TRAIT
#undef CUSTOM_CASE_TO_COMPARE_DTYPE

}  // namespace custom
