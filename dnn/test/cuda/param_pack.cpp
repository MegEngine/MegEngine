/**
 * \file dnn/test/cuda/param_pack.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/checker.h"
#include "test/common/utils.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

namespace {

template <class T>
std::vector<int32_t> create_offsets(const TensorShapeArray& shapes,
                                    size_t alignment) {
    size_t dtype_size = sizeof(T);
    if (alignment < dtype_size)
        alignment = dtype_size;
    alignment /= dtype_size;

    auto get_aligned = [alignment](size_t v) {
        auto mod = v & (alignment - 1);
        return v + ((alignment - mod) & (alignment - 1));
    };

    std::vector<dt_int32> offsets(shapes.size() << 1);
    size_t offset = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
        offset = get_aligned(offset);
        offsets[i << 1] = offset;
        offset += shapes[i].total_nr_elems();
        offsets[(i << 1) + 1] = offset;
    }
    return offsets;
}

template <class T>
std::vector<T> create_pack(size_t pack_size,
                           const std::vector<int32_t>& offsets,
                           const std::vector<std::vector<T>>& ptr) {
    megdnn_assert(pack_size == static_cast<size_t>(offsets.back()));
    std::vector<T> data(pack_size, 0);
    for (size_t i = 0; i * 2 < offsets.size(); ++i) {
        size_t begin = offsets[i * 2], end = offsets[i * 2 + 1];
        for (size_t j = 0; j < end - begin; j++)
            data[begin + j] = ptr[i][j];
    }
    return data;
}

template <class T>
std::vector<std::vector<T>> create_params(size_t nr_params,
                                          const TensorShapeArray& shapes) {
    std::vector<std::vector<T>> params;
    for (size_t i = 0; i < nr_params; ++i) {
        std::vector<T> expected_data;
        for (size_t x = 0; x < shapes[i].total_nr_elems(); ++x) {
            expected_data.push_back(rand());
        }
        params.push_back(std::move(expected_data));
    }
    return params;
}

template <class T>
T* create_device_data(Handle* handle, const T* data, size_t size) {
    T* data_device =
            static_cast<T*>(test::megdnn_malloc(handle, size * sizeof(T)));
    if (data)
        test::megdnn_memcpy_H2D(handle, data_device, data, size * sizeof(T));
    return data_device;
}

template <class T>
void test_param_pack_concat(Handle* handle, const TensorShapeArray& shapes,
                            DType type) {
    auto concat = handle->create_operator<ParamPackConcat>();
    size_t nr_params = shapes.size();

    std::vector<T*> param_ptrs;
    std::vector<std::vector<T>> params = create_params<T>(nr_params, shapes);
    for (size_t i = 0; i < nr_params; ++i) {
        param_ptrs.push_back(create_device_data<T>(handle, params[i].data(),
                                                   shapes[i].total_nr_elems()));
    }
    std::vector<int32_t> offsets =
            create_offsets<T>(shapes, handle->alignment_requirement());
    size_t pack_size = offsets.back();
    int32_t* offsets_gpu =
            create_device_data<int32_t>(handle, offsets.data(), offsets.size());

    std::vector<T> expected_pack = create_pack<T>(pack_size, offsets, params);
    T* pack_gpu = create_device_data<T>(handle, nullptr, expected_pack.size());

    TensorLayout dst_layout({pack_size}, type);
    TensorND dst_tensor(pack_gpu, dst_layout);

    TensorLayout offsets_layout({offsets.size()}, dtype::Int32());
    TensorND offsets_tensor(offsets_gpu, offsets_layout);

    test::WorkspaceWrapper workspace(
            handle, concat->get_workspace_in_bytes(shapes, offsets_layout,
                                                   {pack_size}));
    TensorND src_tensor(param_ptrs.data(),
                        TensorLayout({nr_params}, dtype::Int32()));

    concat->exec(src_tensor, offsets_tensor, dst_tensor, workspace.workspace());

    // check
    T* actual_pack = static_cast<T*>(malloc(pack_size * sizeof(T)));
    test::megdnn_memcpy_D2H(handle, actual_pack, pack_gpu,
                            sizeof(T) * pack_size);
    for (size_t i = 0; i < pack_size; ++i) {
        ASSERT_EQ(actual_pack[i], expected_pack[i]);
    }
    free(actual_pack);
    test::megdnn_free(handle, pack_gpu);
    test::megdnn_free(handle, offsets_gpu);
    for (auto ptr : param_ptrs) {
        test::megdnn_free(handle, ptr);
    }
}

}  // namespace

TEST_F(CUDA, PARAM_PACK) {
    SmallVector<TensorShapeArray> shapes_vec;
    shapes_vec.push_back({{1}});
    shapes_vec.push_back({{129}, {21}});
    shapes_vec.push_back({{15}, {21}, {34}});
    shapes_vec.push_back({{1, 2}, {3, 5}, {5, 8}, {7, 11}, {9, 14}});
    shapes_vec.push_back({{1, 2},
                          {3, 5},
                          {1},
                          {3, 3, 3, 4},
                          {71},
                          {9, 14},
                          {111, 111, 111},
                          {128, 128, 128}});
    for (auto shapes : shapes_vec) {
        test_param_pack_concat<int32_t>(handle_cuda(), shapes, dtype::Int32());
        test_param_pack_concat<int16_t>(handle_cuda(), shapes, dtype::Int16());
        test_param_pack_concat<float>(handle_cuda(), shapes, dtype::Float32());
    }
}

// vim: syntax=cpp.doxygen
