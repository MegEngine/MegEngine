#pragma once

#include <cstddef>
#include <cstdint>

namespace custom {

#ifdef __CUDACC__
#define CUSTOM_HOST   __host__
#define CUSTOM_DEVICE __device__
#else
#define CUSTOM_HOST
#define CUSTOM_DEVICE
#endif

#define CUSTOM_HOST_DEVICE CUSTOM_HOST CUSTOM_DEVICE

template <typename T>
struct DefaultPtrTraits {
    using PtrType = T*;
};

#ifdef __CUDACC__
template <typename T>
struct RestrictPtrTraits {
    using PtrType = T* __restrict__;
};
#endif

template <
        typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits,
        typename index_t = int64_t>
class TensorAccessorProxyBase {
public:
    using PtrType = typename PtrTraits<T>::PtrType;

protected:
    PtrType m_data;
    const index_t* m_sizes;
    const index_t* m_strides;

public:
    CUSTOM_HOST_DEVICE TensorAccessorProxyBase(
            PtrType data, const index_t* sizes, const index_t* strides) {
        m_data = data;
        m_sizes = sizes;
        m_strides = strides;
    }

    CUSTOM_HOST_DEVICE index_t stride(index_t i) const { return m_strides[i]; }

    CUSTOM_HOST_DEVICE index_t size(index_t i) const { return m_sizes[i]; }

    CUSTOM_HOST_DEVICE PtrType data() const { return m_data; }
};

template <
        typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits,
        typename index_t = int64_t>
class TensorAccessorProxy : public TensorAccessorProxyBase<T, N, PtrTraits, index_t> {
public:
    using PtrType = typename PtrTraits<T>::PtrType;

    CUSTOM_HOST_DEVICE TensorAccessorProxy(
            PtrType data, const index_t* sizes, const index_t* strides)
            : TensorAccessorProxyBase<T, N, PtrTraits, index_t>(data, sizes, strides) {}

    CUSTOM_HOST_DEVICE TensorAccessorProxy<T, N - 1, PtrTraits, index_t> operator[](
            index_t i) {
        return TensorAccessorProxy<T, N - 1, PtrTraits, index_t>(
                this->m_data + this->m_strides[0] * i, this->m_sizes + 1,
                this->m_strides + 1);
    }

    CUSTOM_HOST_DEVICE const TensorAccessorProxy<T, N - 1, PtrTraits, index_t>
    operator[](index_t i) const {
        return TensorAccessorProxy<T, N - 1, PtrTraits, index_t>(
                this->m_data + this->m_strides[0] * i, this->m_sizes + 1,
                this->m_strides + 1);
    }
};

template <typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessorProxy<T, 1, PtrTraits, index_t>
        : public TensorAccessorProxyBase<T, 1, PtrTraits, index_t> {
public:
    using PtrType = typename PtrTraits<T>::PtrType;

    CUSTOM_HOST_DEVICE TensorAccessorProxy(
            PtrType data, const index_t* sizes, const index_t* strides)
            : TensorAccessorProxyBase<T, 1, PtrTraits, index_t>(data, sizes, strides) {}

    CUSTOM_HOST_DEVICE T& operator[](index_t i) {
        return this->m_data[this->m_strides[0] * i];
    }

    CUSTOM_HOST_DEVICE const T& operator[](index_t i) const {
        return this->m_data[this->m_strides[0] * i];
    }
};

template <
        typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits,
        typename index_t = int64_t>
class TensorAccessorBase {
public:
    using PtrType = typename PtrTraits<T>::PtrType;

protected:
    PtrType m_data;
    index_t m_sizes[N];
    index_t m_strides[N];

public:
    CUSTOM_HOST_DEVICE TensorAccessorBase(
            PtrType data, const size_t* sizes, const ptrdiff_t* strides) {
        m_data = data;
        for (size_t i = 0; i < N; ++i) {
            m_sizes[i] = sizes[i];
            m_strides[i] = strides[i];
        }
    }

    CUSTOM_HOST_DEVICE index_t stride(index_t i) const { return m_strides[i]; }

    CUSTOM_HOST_DEVICE index_t size(index_t i) const { return m_sizes[i]; }

    CUSTOM_HOST_DEVICE PtrType data() const { return m_data; }
};

template <
        typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits,
        typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T, N, PtrTraits, index_t> {
public:
    using PtrType = typename PtrTraits<T>::PtrType;

    CUSTOM_HOST_DEVICE TensorAccessor(
            PtrType data, const size_t* sizes, const ptrdiff_t* strides)
            : TensorAccessorBase<T, N, PtrTraits, index_t>(data, sizes, strides) {}

    CUSTOM_HOST_DEVICE decltype(auto) operator[](index_t i) {
        return TensorAccessorProxy<T, N, PtrTraits, index_t>(
                this->m_data, this->m_sizes, this->m_strides)[i];
    }

    CUSTOM_HOST_DEVICE decltype(auto) operator[](index_t i) const {
        return TensorAccessorProxy<T, N, PtrTraits, index_t>(
                this->m_data, this->m_sizes, this->m_strides)[i];
    }
};

}  // namespace custom
