#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "../src/misc.h"
#include "lite-c/global_c.h"
#include "lite-c/tensor_c.h"

#include <gtest/gtest.h>
#include <memory>
#include <thread>

TEST(TestCapiTensor, Basic) {
    LiteTensor c_tensor0, c_tensor1;
    LiteTensorDesc description = default_desc;
    LITE_make_tensor(description, &c_tensor0);
    int is_pinned_host = false;
    LITE_is_pinned_host(c_tensor0, &is_pinned_host);
    ASSERT_FALSE(is_pinned_host);
    LiteDeviceType device_type;
    LITE_get_tensor_device_type(c_tensor0, &device_type);
    ASSERT_EQ(device_type, LiteDeviceType::LITE_CPU);
    size_t length = 0;
    LITE_get_tensor_total_size_in_byte(c_tensor0, &length);
    ASSERT_EQ(length, 0);

    LiteLayout layout{{1, 3, 224, 224}, 4, LiteDataType::LITE_FLOAT};
    description.device_type = LiteDeviceType::LITE_CPU;
    description.layout = layout;
    description.is_pinned_host = true;
    LITE_make_tensor(description, &c_tensor1);
    LITE_is_pinned_host(c_tensor1, &is_pinned_host);
    ASSERT_TRUE(is_pinned_host);
    LITE_get_tensor_total_size_in_byte(c_tensor1, &length);
    ASSERT_EQ(length, 1 * 3 * 224 * 224 * 4);

    LiteLayout get_layout;
    LITE_get_tensor_layout(c_tensor1, &get_layout);
    ASSERT_EQ(get_layout.ndim, layout.ndim);
    ASSERT_EQ(get_layout.data_type, layout.data_type);
    ASSERT_EQ(get_layout.shapes[0], layout.shapes[0]);
    ASSERT_EQ(get_layout.shapes[1], layout.shapes[1]);
    ASSERT_EQ(get_layout.shapes[2], layout.shapes[2]);
    ASSERT_EQ(get_layout.shapes[3], layout.shapes[3]);

    //! test error
    ASSERT_EQ(LITE_is_pinned_host(c_tensor0, nullptr), -1);
    ASSERT_NE(strlen(LITE_get_last_error()), 0);
    ASSERT_EQ(LITE_get_last_error_code(), ErrorCode::LITE_INTERNAL_ERROR);
    printf("The last error is: %s\n", LITE_get_last_error());
    LITE_clear_last_error();
    ASSERT_EQ(strlen(LITE_get_last_error()), 0);
    ASSERT_EQ(LITE_get_last_error_code(), ErrorCode::OK);

    LITE_destroy_tensor(c_tensor0);
    LITE_destroy_tensor(c_tensor1);
}

TEST(TestCapiTensor, SetLayoutReAlloc) {
    LiteTensor c_tensor0;
    LiteTensorDesc description = default_desc;
    description.layout = LiteLayout{{1, 3, 224, 224}, 4, LiteDataType::LITE_FLOAT};
    LITE_make_tensor(description, &c_tensor0);
    void *old_ptr, *new_ptr;
    LITE_get_tensor_memory(c_tensor0, &old_ptr);

    LiteLayout new_layout = LiteLayout{{1, 3, 100, 100}, 4, LiteDataType::LITE_INT8};
    LITE_set_tensor_layout(c_tensor0, new_layout);
    LITE_get_tensor_memory(c_tensor0, &new_ptr);

    size_t length = 0;
    LITE_get_tensor_total_size_in_byte(c_tensor0, &length);

    ASSERT_EQ(length, 1 * 3 * 100 * 100);
    ASSERT_EQ(old_ptr, new_ptr);
}

TEST(TestCapiTensor, Reset) {
    LiteTensor c_tensor0, c_tensor1;
    LiteTensorDesc description = default_desc;
    description.layout = LiteLayout{{3, 20}, 2, LiteDataType::LITE_FLOAT};
    LITE_make_tensor(description, &c_tensor0);
    LITE_make_tensor(description, &c_tensor1);
    void *old_ptr0, *old_ptr1;
    LITE_get_tensor_memory(c_tensor0, &old_ptr0);
    LITE_get_tensor_memory(c_tensor1, &old_ptr1);
    //! make sure memory is allocted
    ASSERT_NO_THROW(memcpy(old_ptr0, old_ptr1, 3 * 20 * 4));

    std::shared_ptr<float> new_ptr0(
            new float[3 * 20], [](float* ptr) { delete[] ptr; });
    std::shared_ptr<float> new_ptr1(
            new float[3 * 20], [](float* ptr) { delete[] ptr; });
    LITE_reset_tensor_memory(c_tensor0, new_ptr0.get(), 3 * 20 * 4);
    LITE_reset_tensor_memory(c_tensor1, new_ptr1.get(), 3 * 20 * 4);
    void *tmp_ptr0, *tmp_ptr1;
    LITE_get_tensor_memory(c_tensor0, &tmp_ptr0);
    LITE_get_tensor_memory(c_tensor1, &tmp_ptr1);
    ASSERT_EQ(tmp_ptr0, new_ptr0.get());
    ASSERT_EQ(tmp_ptr1, new_ptr1.get());

    ASSERT_NO_THROW(memcpy(new_ptr0.get(), new_ptr1.get(), 3 * 20 * 4));

    LiteLayout layout1{{6, 20}, 2, LiteDataType::LITE_FLOAT};
    std::shared_ptr<float> ptr2(new float[6 * 20], [](float* ptr) { delete[] ptr; });
    std::shared_ptr<float> ptr3(new float[6 * 20], [](float* ptr) { delete[] ptr; });
    LITE_reset_tensor(c_tensor0, layout1, new_ptr0.get());
    LITE_reset_tensor(c_tensor1, layout1, new_ptr1.get());

    //! memory is not freed by Tensor reset
    ASSERT_NO_THROW(memcpy(new_ptr0.get(), new_ptr1.get(), 3 * 20 * 4));

    LiteLayout tmp_layout0, tmp_layout1;
    LITE_get_tensor_layout(c_tensor0, &tmp_layout0);
    LITE_get_tensor_layout(c_tensor1, &tmp_layout1);
    ASSERT_EQ(tmp_layout0.ndim, tmp_layout1.ndim);
    ASSERT_EQ(tmp_layout0.data_type, tmp_layout1.data_type);
    ASSERT_EQ(tmp_layout0.shapes[0], tmp_layout1.shapes[0]);
    ASSERT_EQ(tmp_layout0.shapes[1], tmp_layout1.shapes[1]);

    LITE_destroy_tensor(c_tensor0);
    LITE_destroy_tensor(c_tensor1);
}

TEST(TestCapiTensor, CrossCNCopy) {
    LiteTensor c_tensor0, c_tensor1, c_tensor2;
    LiteTensorDesc description = default_desc;
    LITE_make_tensor(description, &c_tensor0);

    description.layout = LiteLayout{{1, 3, 224, 224}, 4, LiteDataType::LITE_FLOAT};
    LITE_make_tensor(description, &c_tensor1);
    LITE_make_tensor(description, &c_tensor2);

    LITE_tensor_copy(c_tensor1, c_tensor2);
    LITE_tensor_copy(c_tensor2, c_tensor1);
    void *old_ptr1, *old_ptr2, *new_ptr1, *new_ptr2;
    LITE_get_tensor_memory(c_tensor1, &old_ptr1);
    LITE_get_tensor_memory(c_tensor2, &old_ptr2);

    //! test source tenor is empty
    ASSERT_EQ(LITE_tensor_copy(c_tensor1, c_tensor0), -1);
    ASSERT_NE(strlen(LITE_get_last_error()), 0);
    printf("The last error is: %s\n", LITE_get_last_error());

    LITE_tensor_copy(c_tensor0, c_tensor1);
    LITE_tensor_copy(c_tensor1, c_tensor2);
    LITE_tensor_copy(c_tensor2, c_tensor0);

    LITE_get_tensor_memory(c_tensor1, &new_ptr1);
    LITE_get_tensor_memory(c_tensor2, &new_ptr2);

    ASSERT_EQ(old_ptr1, new_ptr1);
    ASSERT_EQ(old_ptr2, new_ptr2);

    LITE_destroy_tensor(c_tensor0);
    LITE_destroy_tensor(c_tensor1);
    LITE_destroy_tensor(c_tensor2);
}

TEST(TestCapiTensor, ShareMemoryWith) {
    LiteTensor c_tensor0, c_tensor1;
    LiteTensorDesc description = default_desc;
    LITE_make_tensor(description, &c_tensor0);

    description.layout = LiteLayout{{1, 3, 224, 224}, 4, LiteDataType::LITE_FLOAT};
    LITE_make_tensor(description, &c_tensor1);

    ASSERT_EQ(LITE_tensor_share_memory_with(c_tensor1, c_tensor0), -1);
    LITE_tensor_share_memory_with(c_tensor0, c_tensor1);
    void *ptr0, *ptr1;
    LITE_get_tensor_memory(c_tensor0, &ptr0);
    LITE_get_tensor_memory(c_tensor1, &ptr1);

    ASSERT_EQ(ptr0, ptr1);

    LITE_destroy_tensor(c_tensor0);
    LITE_destroy_tensor(c_tensor1);
}

TEST(TestCapiTensor, Reshape) {
    LiteTensor c_tensor0;
    LiteTensorDesc description = default_desc;
    description.layout = LiteLayout{{8, 8, 100, 100}, 4, LiteDataType::LITE_FLOAT};
    LITE_make_tensor(description, &c_tensor0);
    void* old_ptr;
    LITE_get_tensor_memory(c_tensor0, &old_ptr);

    auto check = [&](std::vector<size_t> expect, const LiteTensor& tensor) {
        LiteLayout get_layout;
        LITE_get_tensor_layout(tensor, &get_layout);
        ASSERT_EQ(get_layout.ndim, expect.size());
        for (size_t i = 0; i < expect.size(); i++) {
            ASSERT_EQ(get_layout.shapes[i], expect[i]);
        }
        void* new_ptr;
        LITE_get_tensor_memory(tensor, &new_ptr);
        ASSERT_EQ(old_ptr, new_ptr);
    };
    {
        int shape[2] = {-1, 50};
        LITE_tensor_reshape(c_tensor0, shape, 2);
        check({8 * 8 * 100 * 2, 50}, c_tensor0);
    }
    {
        int shape[3] = {64, 100, 100};
        LITE_tensor_reshape(c_tensor0, shape, 3);
        check({8 * 8, 100, 100}, c_tensor0);
    }
    {
        int shape[3] = {16, 100, -1};
        LITE_tensor_reshape(c_tensor0, shape, 3);
        check({16, 100, 400}, c_tensor0);
    }
    LITE_destroy_tensor(c_tensor0);
}

TEST(TestCapiTensor, Slice) {
    LiteTensor c_tensor0;
    LiteTensorDesc description = default_desc;
    description.layout = LiteLayout{{20, 20}, 2, LiteDataType::LITE_FLOAT};
    LITE_make_tensor(description, &c_tensor0);
    void* old_ptr;
    LITE_get_tensor_memory(c_tensor0, &old_ptr);
    for (size_t i = 0; i < 20 * 20; i++) {
        *(static_cast<float*>(old_ptr) + i) = i;
    }
    auto check = [&](size_t start, size_t end, size_t step, bool have_step) {
        LiteTensor tensor, slice_tensor;
        LITE_make_tensor(default_desc, &tensor);
        size_t start_ptr[2] = {start, start};
        size_t end_ptr[2] = {end, end};
        size_t step_ptr[2] = {step, step};

        if (have_step) {
            LITE_tensor_slice(
                    c_tensor0, start_ptr, end_ptr, step_ptr, 2, &slice_tensor);
        } else {
            LITE_tensor_slice(c_tensor0, start_ptr, end_ptr, nullptr, 2, &slice_tensor);
        }
        int is_continue = true;
        LITE_is_memory_continue(slice_tensor, &is_continue);
        ASSERT_FALSE(is_continue);

        LITE_tensor_copy(tensor, slice_tensor);
        void* new_ptr;
        LITE_get_tensor_memory(tensor, &new_ptr);
        float* ptr = static_cast<float*>(new_ptr);
        for (size_t i = start; i < end; i += step) {
            for (size_t j = start; j < end; j += step) {
                ASSERT_EQ(float(i * 20 + j), *ptr);
                ++ptr;
            }
        }
        LITE_destroy_tensor(tensor);
        LITE_destroy_tensor(slice_tensor);
    };
    check(1, 8, 1, true);
    check(1, 8, 1, false);
    check(2, 10, 2, true);
    check(10, 18, 4, true);
    check(10, 18, 1, false);
    LITE_destroy_tensor(c_tensor0);
}

TEST(TestCapiTensor, Memset) {
    LiteTensor c_tensor0;
    LiteTensorDesc description = default_desc;
    description.layout = LiteLayout{{20, 20}, 2, LiteDataType::LITE_FLOAT};
    LITE_make_tensor(description, &c_tensor0);
    void* ptr;
    uint8_t* uint8_ptr;
    LITE_get_tensor_memory(c_tensor0, &ptr);
    LITE_tensor_fill_zero(c_tensor0);
    uint8_ptr = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < 20 * 20; i++) {
        ASSERT_EQ(0, *uint8_ptr);
        uint8_ptr++;
    }

    LITE_destroy_tensor(c_tensor0);
}

TEST(TestCapiTensor, GetMemoryByIndex) {
    LiteTensor c_tensor0;
    LiteTensorDesc description = default_desc;
    description.layout = LiteLayout{{20, 20}, 2, LiteDataType::LITE_FLOAT};
    LITE_make_tensor(description, &c_tensor0);
    void *ptr0, *ptr1, *ptr2, *ptr3;
    LITE_get_tensor_memory(c_tensor0, &ptr0);
    size_t index0[] = {3, 4};
    LITE_get_tensor_memory_with_index(c_tensor0, &index0[0], 2, &ptr1);
    size_t index1[] = {5, 7};
    LITE_get_tensor_memory_with_index(c_tensor0, &index1[0], 2, &ptr2);
    size_t index2[] = {5};
    LITE_get_tensor_memory_with_index(c_tensor0, &index2[0], 1, &ptr3);

    ASSERT_EQ(ptr1, static_cast<float*>(ptr0) + 3 * 20 + 4);
    ASSERT_EQ(ptr2, static_cast<float*>(ptr0) + 5 * 20 + 7);
    ASSERT_EQ(ptr3, static_cast<float*>(ptr0) + 5 * 20);

    LITE_destroy_tensor(c_tensor0);
}

TEST(TestCapiTensor, ThreadLocalError) {
    LiteTensor c_tensor0;
    LiteTensorDesc description = default_desc;
    description.layout = LiteLayout{{20, 20}, 2, LiteDataType::LITE_FLOAT};
    void *ptr0, *ptr1;
    std::thread thread1([&]() {
        LITE_make_tensor(description, &c_tensor0);
        LITE_get_tensor_memory(c_tensor0, &ptr0);
    });
    thread1.join();
    std::thread thread2([&]() {
        LITE_get_tensor_memory(c_tensor0, &ptr1);
        LITE_destroy_tensor(c_tensor0);
    });
    thread2.join();
}

TEST(TestCapiTensor, GlobalHolder) {
    LiteTensor c_tensor0;
    LiteTensorDesc description = default_desc;
    description.layout = LiteLayout{{20, 20}, 2, LiteDataType::LITE_FLOAT};

    LITE_make_tensor(description, &c_tensor0);
    auto destroy_tensor = c_tensor0;

    LITE_make_tensor(description, &c_tensor0);
    //! make sure destroy_tensor is destroyed by LITE_make_tensor
    LITE_destroy_tensor(destroy_tensor);
    ASSERT_EQ(LITE_destroy_tensor(destroy_tensor), 0);
    LITE_destroy_tensor(c_tensor0);
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
