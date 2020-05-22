/**
 * \file dnn/src/common/cv/mat.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"

#ifdef MEGDNN_CC_CUDA
#include "src/cuda/utils.cuh"
#endif

namespace megdnn {
namespace megcv {

#ifdef MEGDNN_CC_CUDA

template <typename T>
Mat<T>::Mat(size_t rows, size_t cols, size_t channels, size_t step)
        : m_rows(rows),
          m_cols(cols),
          m_channels(channels),
          m_step(step),
          m_offset(0) {
    megdnn_assert(step >= cols * channels);
    megdnn_assert(1 <= channels && channels <= 4);
    T* raw_data;
    cuda_check(cudaMalloc((void**)&raw_data, sizeof(T) * rows * step));
    m_data =
            std::shared_ptr<T>(raw_data, [](T* d) { cuda_check(cudaFree(d)); });
    cudaMemset(m_data.get(), 0, sizeof(T) * rows * step);
}

template <typename T>
Mat<T>::Mat(size_t rows, size_t cols, size_t channels)
        : Mat(rows, cols, channels, cols * channels) {}

template <typename T>
Mat<T>::Mat(size_t rows, size_t cols, size_t channels, T* data)
        : m_rows(rows),
          m_cols(cols),
          m_channels(channels),
          m_step(cols * channels),
          m_data(data, [](T*) {}),
          m_offset(0) {}

template <typename T>
Mat<T>::Mat(const Mat<T>& rhs)
        : m_rows(rhs.m_rows),
          m_cols(rhs.m_cols),
          m_channels(rhs.m_channels),
          m_step(rhs.m_step),
          m_data(rhs.m_data),
          m_offset(0) {}

template <typename T>
Mat<T>::Mat(const Mat<T>& rhs, size_t row_offset, size_t row_count,
            size_t col_offset, size_t col_count)
        : m_rows(row_count),
          m_cols(col_count),
          m_channels(rhs.m_channels),
          m_step(rhs.m_step),
          m_data(rhs.m_data),
          m_offset(rhs.m_offset + row_offset * m_step +
                   col_offset * m_channels) {}

template <typename T>
Mat<T>& Mat<T>::operator=(const Mat<T>& rhs) {
    this->m_rows = rhs.m_rows;
    this->m_cols = rhs.m_cols;
    this->m_channels = rhs.m_channels;
    this->m_step = rhs.m_step;
    this->m_data = rhs.m_data;
    this->m_offset = rhs.m_offset;
    return *this;
}

template <typename T>
T& Mat<T>::at(size_t r, size_t c, size_t ch) {
    megdnn_assert(r < m_rows);
    megdnn_assert(c < m_cols);
    megdnn_assert(ch < m_channels);
    return ptr(r)[c * m_channels + ch];
}

template <typename T>
const T& Mat<T>::at(size_t r, size_t c, size_t ch) const {
    megdnn_assert(r < m_rows);
    megdnn_assert(c < m_cols);
    megdnn_assert(ch < m_channels);
    return ptr(r)[c * m_channels + ch];
}

template <typename T>
Mat<T> Mat<T>::clone() const {
    Mat<T> res(m_rows, m_cols, m_channels);
    for (size_t r = 0; r < m_rows; ++r) {
        cuda_check(cudaMemcpy(res.ptr(r), this->ptr(r),
                              sizeof(T) * m_cols * m_channels,
                              cudaMemcpyDeviceToDevice));
    }
    return res;
}

template <typename T>
bool Mat<T>::equals(const Mat<T>& rhs) const {
    if (this->m_rows != rhs.m_rows)
        return false;
    if (this->m_cols != rhs.m_cols)
        return false;
    if (this->m_channels != rhs.m_channels)
        return false;
    std::unique_ptr<T[]> row1(new T[m_cols * m_channels]);
    std::unique_ptr<T[]> row2(new T[m_cols * m_channels]);
    megdnn_assert(row1);
    megdnn_assert(row2);
    for (size_t r = 0; r < m_rows; ++r) {
        cuda_check(cudaMemcpy(row1.get(), this->ptr(r),
                              sizeof(T) * m_cols * m_channels,
                              cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(row2.get(), rhs.ptr(r), sizeof(T) * m_cols * m_channels,
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < m_cols * m_channels; ++i) {
            if (row1[i] != row2[i])
                return false;
        }
    }
    return true;
}

template <typename T>
bool Mat<T>::is_continuous() const {
    return m_step == m_cols * m_channels;
}

template <typename T>
void Mat<T>::read(const T* src) {
    megdnn_assert(is_continuous());
    cuda_check(cudaMemcpy(m_data.get(), src, sizeof(T) * this->total_nr_elem(),
                          cudaMemcpyHostToDevice));
}

template <typename T>
void Mat<T>::write(T* dst) const {
    megdnn_assert(is_continuous());
    cuda_check(cudaMemcpy(dst, m_data.get(), sizeof(T) * this->total_nr_elem(),
                          cudaMemcpyDeviceToHost));
}

template class Mat<uchar>;
template class Mat<float>;
template class Mat<double>;
template class Mat<short>;
template class Mat<unsigned short>;

#else

template <typename T>
Mat<T>::Mat()
        : m_rows(0),
          m_cols(0),
          m_channels(0),
          m_step(0),
          m_data(nullptr),
          m_offset(0) {}

template <typename T>
Mat<T>::Mat(size_t rows, size_t cols, size_t channels, size_t step)
        : m_rows(rows),
          m_cols(cols),
          m_channels(channels),
          m_step(step),
          m_data(new T[rows * step], [](T* d) { delete[] d; }),
          m_offset(0) {
    megdnn_assert(step >= cols * channels);
    megdnn_assert(1 <= channels && channels <= 4);
    memset(m_data.get(), 0, sizeof(T) * rows * step);
}

template <typename T>
Mat<T> TensorND2Mat(const TensorND& tensor, size_t batch) {
    size_t m_rows = tensor.layout.shape[1];
    size_t m_cols = tensor.layout.shape[2];
    size_t m_channels = tensor.layout.shape[3];
    size_t m_step = tensor.layout.stride[1];
    T* data = ((T*)tensor.ptr<T>()) + m_step * m_rows * batch;

    Mat<T> mat(m_rows, m_cols, m_channels, m_step, data);
    return mat;
}

template <>
Mat<int> TensorND2Mat<int>(const TensorND& tensor, size_t batch) {
    size_t m_rows = tensor.layout.shape[1];
    size_t m_cols = tensor.layout.shape[2];
    size_t m_channels = tensor.layout.shape[3];
    size_t m_step = tensor.layout.stride[1];

    int* data = tensor.ptr<int>() + m_step * m_rows * batch;

    Mat<int> mat(m_rows, m_cols, m_channels, m_step, data);
    return mat;
}

template <>
Mat<float> TensorND2Mat<float>(const TensorND& tensor, size_t batch) {
    size_t m_rows = tensor.layout.shape[1];
    size_t m_cols = tensor.layout.shape[2];
    size_t m_channels = tensor.layout.shape[3];
    size_t m_step = tensor.layout.stride[1];
    float* data = tensor.ptr<float>() + m_step * m_rows * batch;
    // m_data = std::shared_ptr<T>(data, [](T *) {});

    Mat<float> mat(m_rows, m_cols, m_channels, m_step, data);
    return mat;
}

template <>
Mat<uchar> TensorND2Mat<uchar>(const TensorND& tensor, size_t batch) {
    size_t m_rows = tensor.layout.shape[1];
    size_t m_cols = tensor.layout.shape[2];
    size_t m_channels = tensor.layout.shape[3];
    size_t m_step = tensor.layout.stride[1];
    uchar* data = tensor.ptr<uchar>() + m_step * m_rows * batch;
    // m_data = std::shared_ptr<T>(data, [](T *) {});

    Mat<uchar> mat(m_rows, m_cols, m_channels, m_step, data);
    return mat;
}

template <typename T>
Mat<T>::Mat(size_t rows, size_t cols, size_t channels)
        : Mat(rows, cols, channels, cols * channels) {}

template <typename T>
Mat<T>::Mat(size_t rows, size_t cols, size_t channels, T* data)
        : m_rows(rows),
          m_cols(cols),
          m_channels(channels),
          m_step(cols * channels),
          m_data(data, [](T*) {}),
          m_offset(0) {}

template <typename T>
Mat<T>::Mat(size_t rows, size_t cols, size_t channels, size_t step, T* data)
        : m_rows(rows),
          m_cols(cols),
          m_channels(channels),
          m_step(step),
          m_data(data, [](T*) {}),
          m_offset(0) {}

template <typename T>
Mat<T>::Mat(const Mat<T>& rhs)
        : m_rows(rhs.m_rows),
          m_cols(rhs.m_cols),
          m_channels(rhs.m_channels),
          m_step(rhs.m_step),
          m_data(rhs.m_data),
          m_offset(0) {}

template <typename T>
Mat<T>::Mat(const Mat<T>& rhs, size_t row_offset, size_t row_count,
            size_t col_offset, size_t col_count)
        : m_rows(row_count),
          m_cols(col_count),
          m_channels(rhs.m_channels),
          m_step(rhs.m_step),
          m_data(rhs.m_data),
          m_offset(rhs.m_offset + row_offset * m_step +
                   col_offset * m_channels) {}

template <typename T>
Mat<T>& Mat<T>::operator=(const Mat<T>& rhs) {
    this->m_rows = rhs.m_rows;
    this->m_cols = rhs.m_cols;
    this->m_channels = rhs.m_channels;
    this->m_step = rhs.m_step;
    this->m_data = rhs.m_data;
    this->m_offset = rhs.m_offset;
    return *this;
}

template <typename T>
T& Mat<T>::at(size_t r, size_t c, size_t ch) {
    megdnn_assert(r < m_rows);
    megdnn_assert(c < m_cols);
    megdnn_assert(ch < m_channels);
    return ptr(r)[c * m_channels + ch];
}

template <typename T>
const T& Mat<T>::at(size_t r, size_t c, size_t ch) const {
    megdnn_assert(r < m_rows);
    megdnn_assert(c < m_cols);
    megdnn_assert(ch < m_channels);
    return ptr(r)[c * m_channels + ch];
}

template <typename T>
Mat<T> Mat<T>::clone() const {
    Mat<T> res(m_rows, m_cols, m_channels);
    for (size_t r = 0; r < m_rows; ++r) {
        memcpy(res.ptr(r), this->ptr(r), sizeof(T) * m_cols * m_channels);
    }
    return res;
}

template <typename T>
bool Mat<T>::equals(const Mat<T>& rhs) const {
    if (this->m_rows != rhs.m_rows)
        return false;
    if (this->m_cols != rhs.m_cols)
        return false;
    if (this->m_channels != rhs.m_channels)
        return false;
    for (size_t r = 0; r < m_rows; ++r) {
        if (0 !=
            memcmp(this->ptr(r), rhs.ptr(r), sizeof(T) * m_cols * m_channels))
            return false;
    }
    return true;
}

template <typename T>
bool Mat<T>::is_continuous() const {
    return m_step == m_cols * m_channels;
}

template <typename T>
void Mat<T>::read(const T* src) {
    megdnn_assert(is_continuous());
    memcpy(m_data.get(), src, sizeof(T) * this->total_nr_elem());
}

template <typename T>
void Mat<T>::write(T* dst) const {
    megdnn_assert(is_continuous());
    memcpy(dst, m_data.get(), sizeof(T) * this->total_nr_elem());
}

template class Mat<uchar>;
template class Mat<float>;
template class Mat<double>;
template class Mat<short>;
template class Mat<unsigned short>;
template class Mat<int>;

#endif

}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
