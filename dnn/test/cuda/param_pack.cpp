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

template<class T>
std::vector<int32_t> create_table(const TensorShapeArray& shapes,
                  size_t align) {
    size_t dtype_size = sizeof(T);
    if (align < dtype_size)
        align = dtype_size;

    align /= dtype_size;

    size_t offset = shapes[0].total_nr_elems();
    for (size_t i = 1; i < shapes.size(); i++) {
        auto d = offset & (align - 1);
        offset += (align - d) & (align - 1);

        offset += shapes[i].total_nr_elems();
    }

    std::vector<int32_t> table(offset * 2);

    int32_t* outer_table = table.data();
    int32_t* inner_table = outer_table + offset;

    offset = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
        for (; (offset & (align - 1)) != 0; offset++) {
            outer_table[offset] = inner_table[offset] = -1;
        }

        size_t j = 0;
        for (; j < shapes[i].total_nr_elems(); j++) {
            outer_table[offset + j] = i;
            inner_table[offset + j] = j;
        }
        offset += j;
    }
    return table;
}

template<class T>
std::vector<T> create_pack(size_t pack_size, const std::vector<int32_t>& table,
        const std::vector<std::vector<T>>& ptr) {
    assert(pack_size == table.size() / 2);
    const int32_t* outer_table = table.data();
    const int32_t* inner_table = outer_table + pack_size;
    std::vector<T> data(pack_size);
    for (size_t idx = 0; idx < pack_size; ++idx) {
        int32_t out_idx = outer_table[idx];
        int32_t in_idx = inner_table[idx];
        if (in_idx != -1) {
            data[idx] = ptr[out_idx][in_idx];
        }
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

template<class T>
void test_param_pack_split(Handle* handle, const TensorShapeArray& shapes,
        DType type) {
    auto split = handle->create_operator<ParamPackSplit>();

    size_t nr_params = shapes.size();
    std::vector<T*> param_ptrs;
    for (size_t i = 0; i < nr_params; ++i) {
        param_ptrs.push_back(create_device_data<T>(handle,
                    nullptr, shapes[i].total_nr_elems()));
    }
    std::vector<std::vector<T>> expected_param = create_params<T>(nr_params,
            shapes);

    std::vector<int32_t> table =
            create_table<T>(shapes, handle->alignment_requirement());
    ASSERT_EQ(table,
              ParamPackSplit::gen_table(shapes, handle->alignment_requirement(),
                                        sizeof(T)));
    size_t pack_size = table.size() / 2;
    int32_t* table_gpu = create_device_data<int32_t>(handle, table.data(),
            table.size());

    std::vector<T> pack =
        create_pack<T>(pack_size, table, expected_param);
    T* pack_gpu = create_device_data<T>(handle, pack.data(), pack.size());

    TensorLayout src_layout({pack_size}, type);
    TensorND src_tensor(pack_gpu, src_layout);

    TensorLayout table_layout({table.size()}, dtype::Int32());
    TensorND table_tensor(table_gpu, table_layout);

    test::WorkspaceWrapper workspace(handle, split->get_workspace_in_bytes(
                {pack_size}, table_layout, shapes));
    TensorND dst_tensor(param_ptrs.data(),
            TensorLayout({nr_params}, dtype::Int32()));

    split->exec(src_tensor, table_tensor, dst_tensor, workspace.workspace());


    // check
    for (size_t i = 0; i < nr_params; ++i) {
        T* actual_param = static_cast<T*>(malloc(shapes[i].total_nr_elems()
                    * sizeof(T)));
        test::megdnn_memcpy_D2H(handle, actual_param, param_ptrs[i],
                shapes[i].total_nr_elems() * sizeof(T));
        for (size_t idx = 0; idx < shapes[i].total_nr_elems(); ++idx) {
            ASSERT_EQ(actual_param[idx], expected_param[i][idx]);
        }
        free(actual_param);
    }
    test::megdnn_free(handle, pack_gpu);
    test::megdnn_free(handle, table_gpu);
    for (auto ptr : param_ptrs) {
        test::megdnn_free(handle, ptr);
    }
}

template <class T>
void test_param_pack_concat(Handle* handle, const TensorShapeArray& shapes,
        DType type) {
    auto concat = handle->create_operator<ParamPackConcat>();
    size_t nr_params = shapes.size();

    std::vector<T*> param_ptrs;
    std::vector<std::vector<T>> params = create_params<T>(nr_params,
            shapes);
    for (size_t i = 0; i < nr_params; ++i) {
        param_ptrs.push_back(create_device_data<T>(handle,
                    params[i].data(), shapes[i].total_nr_elems()));
    }
    std::vector<int32_t> table =
            create_table<T>(shapes, handle->alignment_requirement());
    size_t pack_size = table.size() / 2;
    int32_t* table_gpu = create_device_data<int32_t>(handle, table.data(),
            table.size());

    std::vector<T> expected_pack =
        create_pack<T>(pack_size, table, params);
    T* pack_gpu = create_device_data<T>(handle, nullptr, expected_pack.size());

    TensorLayout dst_layout({pack_size}, type);
    TensorND dst_tensor(pack_gpu, dst_layout);

    TensorLayout table_layout({table.size()}, dtype::Int32());
    TensorND table_tensor(table_gpu, table_layout);

    test::WorkspaceWrapper workspace(handle, concat->get_workspace_in_bytes(
                shapes, table_layout, {pack_size}));
    TensorND src_tensor(param_ptrs.data(),
            TensorLayout({nr_params}, dtype::Int32()));

    concat->exec(src_tensor, table_tensor, dst_tensor, workspace.workspace());

    // check
    T* actual_pack = static_cast<T*>(malloc(pack_size * sizeof(T)));
    test::megdnn_memcpy_D2H(handle, actual_pack,
            pack_gpu, sizeof(T) * pack_size);
    for (size_t i = 0; i < pack_size; ++i) {
        ASSERT_EQ(actual_pack[i], expected_pack[i]);
    }
    free(actual_pack);
    test::megdnn_free(handle, pack_gpu);
    test::megdnn_free(handle, table_gpu);
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
        test_param_pack_split<int32_t>(handle_cuda(), shapes, dtype::Int32());
        test_param_pack_split<int16_t>(handle_cuda(), shapes, dtype::Int16());
        test_param_pack_split<float>(handle_cuda(), shapes, dtype::Float32());
        test_param_pack_concat<int32_t>(handle_cuda(), shapes, dtype::Int32());
        test_param_pack_concat<int16_t>(handle_cuda(), shapes, dtype::Int16());
        test_param_pack_concat<float>(handle_cuda(), shapes, dtype::Float32());
    }
}

// vim: syntax=cpp.doxygen
