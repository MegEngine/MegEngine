#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "../src/mge/common.h"
#include "../src/mge/network_impl.h"
#include "../src/misc.h"
#include "lite/tensor.h"

#include <gtest/gtest.h>

#include <string.h>
#include <memory>

using namespace lite;

TEST(TestTensor, Basic) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor1(LiteDeviceType::LITE_CPU);
    Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
    Tensor tensor3(LiteDeviceType::LITE_CPU, layout);
    //! mge tensor has created
    ASSERT_TRUE(TensorHelper::implement(&tensor1));
    ASSERT_TRUE(TensorHelper::implement(&tensor2));
    ASSERT_TRUE(TensorHelper::implement(&tensor3));
    //! check member
    ASSERT_EQ(tensor2.get_device_type(), LiteDeviceType::LITE_CPU);
    ASSERT_EQ(tensor2.get_layout(), layout);
    ASSERT_EQ(tensor3.get_layout(), layout);
    //! check the real tensor
    ASSERT_EQ(tensor2.get_tensor_total_size_in_byte(), 1 * 3 * 224 * 224 * 4);
    ASSERT_EQ(tensor3.get_tensor_total_size_in_byte(), 1 * 3 * 224 * 224 * 4);

    ASSERT_TRUE(TensorHelper::implement(&tensor1)
                        ->cast_final_safe<TensorImplDft>()
                        .host_tensor());

    ASSERT_FALSE(TensorHelper::implement(&tensor1)
                         ->cast_final_safe<TensorImplDft>()
                         .dev_tensor());
    ASSERT_FALSE(TensorHelper::implement(&tensor1)
                         ->cast_final_safe<TensorImplDft>()
                         .dev_tensor());
    ASSERT_TRUE(TensorHelper::implement(&tensor1)
                        ->cast_final_safe<TensorImplDft>()
                        .host_tensor());
}

TEST(TestTensor, SetLayoutReAlloc) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor1;
    Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
    Tensor tensor3(LiteDeviceType::LITE_CPU, layout);
    auto old_ptr2 = tensor2.get_memory_ptr();
    auto old_ptr3 = tensor3.get_memory_ptr();

    //! layout set through
    Layout layout1{{1, 3, 100, 100}, 4, LiteDataType::LITE_INT8};
    tensor1.set_layout(layout1);
    tensor2.set_layout(layout1);
    tensor3.set_layout(layout1);
    ASSERT_EQ(tensor2.get_tensor_total_size_in_byte(), 1 * 3 * 100 * 100);
    ASSERT_EQ(tensor3.get_tensor_total_size_in_byte(), 1 * 3 * 100 * 100);
    auto layout2 = TensorHelper::implement(&tensor2)
                           ->cast_final_safe<TensorImplDft>()
                           .host_tensor()
                           ->layout();
    auto layout3 = TensorHelper::implement(&tensor3)
                           ->cast_final_safe<TensorImplDft>()
                           .host_tensor()
                           ->layout();
    ASSERT_EQ(to_lite_layout(layout2), layout1);
    ASSERT_EQ(to_lite_layout(layout3), layout1);

    auto new_ptr2 = tensor2.get_memory_ptr();
    auto new_ptr3 = tensor3.get_memory_ptr();

    ASSERT_EQ(old_ptr2, new_ptr2);
    ASSERT_EQ(old_ptr3, new_ptr3);
}

TEST(TestTensor, Reset) {
    Layout layout{{3, 20}, 2, LiteDataType::LITE_FLOAT};
    Tensor tensor1;
    Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
    Tensor tensor3(LiteDeviceType::LITE_CPU, layout);

    auto old_ptr2 = tensor2.get_memory_ptr();
    auto old_ptr3 = tensor3.get_memory_ptr();
    //! make sure memory is allocted
    ASSERT_NO_THROW(memcpy(old_ptr2, old_ptr3, 3 * 20 * 2));

    std::shared_ptr<float> new_ptr2(
            new float[3 * 20], [](float* ptr) { delete[] ptr; });
    std::shared_ptr<float> new_ptr3(
            new float[3 * 20], [](float* ptr) { delete[] ptr; });
    tensor1.reset(new_ptr2.get(), layout);
    tensor2.reset(new_ptr2.get(), 3 * 20 * 4);
    tensor3.reset(new_ptr3.get(), 3 * 20 * 4);
    //! After reset the original mem is freed
    /*ASSERT_EXIT((memcpy(old_ptr2, old_ptr3, 3 * 20 * 2), exit(0)),
                ::testing::KilledBySignal(SIGSEGV), ".*");*/

    ASSERT_EQ(tensor2.get_memory_ptr(), new_ptr2.get());
    ASSERT_EQ(tensor3.get_memory_ptr(), new_ptr3.get());

    ASSERT_NO_THROW(memcpy(new_ptr2.get(), new_ptr3.get(), 3 * 20 * 2));

    Layout layout1{{6, 20}, 2, LiteDataType::LITE_FLOAT};
    std::shared_ptr<float> ptr2(new float[6 * 20], [](float* ptr) { delete[] ptr; });
    std::shared_ptr<float> ptr3(new float[6 * 20], [](float* ptr) { delete[] ptr; });
    tensor2.reset(ptr2.get(), layout1);
    tensor3.reset(ptr3.get(), layout1);

    //! memory is not freed by Tensor reset
    ASSERT_NO_THROW(memcpy(new_ptr2.get(), new_ptr3.get(), 3 * 20 * 2));
    auto host_layout2 = TensorHelper::implement(&tensor2)
                                ->cast_final_safe<TensorImplDft>()
                                .host_tensor()
                                ->layout();
    auto host_layout3 = TensorHelper::implement(&tensor3)
                                ->cast_final_safe<TensorImplDft>()
                                .host_tensor()
                                ->layout();

    ASSERT_EQ(to_lite_layout(host_layout2), layout1);
    ASSERT_EQ(to_lite_layout(host_layout3), layout1);
}

TEST(TestTensor, CrossCNCopy) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor1(LiteDeviceType::LITE_CPU);
    Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
    Tensor tensor3(LiteDeviceType::LITE_CPU, layout);
    tensor2.copy_from(tensor3);
    tensor3.copy_from(tensor2);
    auto old_ptr2 = tensor2.get_memory_ptr();
    auto old_ptr3 = tensor3.get_memory_ptr();

    //! test source tenor is empty
    ASSERT_THROW(tensor2.copy_from(tensor1), std::exception);
    tensor1.copy_from(tensor2);
    tensor2.copy_from(tensor3);
    tensor3.copy_from(tensor2);

    ASSERT_EQ(tensor2.get_memory_ptr(), old_ptr2);
    ASSERT_EQ(tensor3.get_memory_ptr(), old_ptr3);
}

TEST(TestTensor, SharedTensorMemory) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor1(LiteDeviceType::LITE_CPU);
    {
        Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
        tensor1.share_memory_with(tensor2);
        auto ptr1 = tensor1.get_memory_ptr();
        auto ptr2 = tensor2.get_memory_ptr();
        ASSERT_EQ(ptr1, ptr2);
    }
    // check after tensor2 destroy, tensor1 can also visit
    auto ptr1 = static_cast<float*>(tensor1.get_memory_ptr());
    size_t length = tensor1.get_tensor_total_size_in_byte() /
                    tensor1.get_layout().get_elem_size();
    for (size_t i = 0; i < length; i++) {
        ptr1[i] = i;
    }
}

TEST(TestTensor, Reshape) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
    auto ptr = tensor2.get_memory_ptr();

    //! test wrong case
    ASSERT_THROW(tensor2.reshape({-1, -1, 3 * 224 * 224}), std::exception);
    ASSERT_THROW(tensor2.reshape({-1, 3, 3 * 224 * 224}), std::exception);
    ASSERT_THROW(tensor2.reshape({1, 3, 3 * 224 * 224}), std::exception);
    ASSERT_THROW(tensor2.reshape({3, 3, 3 * 224 * 224}), std::exception);

    tensor2.reshape({3 * 224 * 224});
    ASSERT_EQ(tensor2.get_layout().ndim, 1);
    ASSERT_EQ(tensor2.get_layout().data_type, LiteDataType::LITE_FLOAT);
    ASSERT_EQ(tensor2.get_layout().shapes[0], 3 * 224 * 224);
    tensor2.reshape({-1, 224, 224});
    ASSERT_EQ(tensor2.get_layout().ndim, 3);
    ASSERT_EQ(tensor2.get_layout().shapes[0], 3);
    ASSERT_EQ(tensor2.get_layout().shapes[1], 224);

    ASSERT_EQ(tensor2.get_memory_ptr(), ptr);
}

TEST(TestTensor, Slice) {
    Layout layout{{20, 20}, 2};
    Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
    auto ptr = tensor2.get_memory_ptr();

    //! test source tenor is empty
    ASSERT_THROW(tensor2.slice({5, 10, 10}, {10, 15}), std::exception);
    ASSERT_THROW(tensor2.slice({5, 10}, {10, 15}, {5}), std::exception);
    ASSERT_THROW(tensor2.slice({5, 10}, {10, 15, 10}), std::exception);
    for (int i = 0; i < 20 * 20; i++) {
        *(static_cast<float*>(ptr) + i) = i;
    }
    auto check = [&](size_t start, size_t end, size_t step) {
        Tensor tensor3;
        tensor3.copy_from(*tensor2.slice({start, start}, {end, end}, {step, step}));
        float* new_ptr = static_cast<float*>(tensor3.get_memory_ptr());
        for (size_t i = start; i < end; i += step) {
            for (size_t j = start; j < end; j += step) {
                ASSERT_EQ(float(i * 20 + j), *new_ptr);
                ++new_ptr;
            }
        }
    };
    check(5, 10, 1);
    check(5, 11, 2);
    check(2, 18, 4);

    Tensor tensor3;
    tensor3.copy_from(*tensor2.slice({3}, {9}, {2}));
    float* new_ptr = static_cast<float*>(tensor3.get_memory_ptr());
    for (size_t i = 3; i < 9; i += 2) {
        for (size_t j = 0; j < 20; j++) {
            ASSERT_EQ(float(i * 20 + j), *new_ptr);
            ++new_ptr;
        }
    }
}

TEST(TestTensor, SliceCopy) {
    Layout layout{{20, 20}, 2};
    Tensor tensor(LiteDeviceType::LITE_CPU, layout);
    //! alloc memory
    auto ptr = static_cast<float*>(tensor.get_memory_ptr());

    Layout layout_slice{{20, 10}, 2};
    Tensor tensor0(LiteDeviceType::LITE_CPU, layout_slice);
    auto ptr0 = tensor0.get_memory_ptr();
    for (int i = 0; i < 10 * 20; i++) {
        *(static_cast<float*>(ptr0) + i) = i;
    }
    Tensor tensor1(LiteDeviceType::LITE_CPU, layout_slice);
    auto ptr1 = tensor1.get_memory_ptr();
    for (int i = 0; i < 10 * 20; i++) {
        *(static_cast<float*>(ptr1) + i) = i + 200;
    }

    auto slice0 = tensor.slice({0, 0}, {20, 10});
    auto slice1 = tensor.slice({0, 10}, {20, 20});

    slice0->copy_from(tensor0);
    slice1->copy_from(tensor1);

    ASSERT_FALSE(slice0->is_continue_memory());
    ASSERT_FALSE(slice1->is_continue_memory());

    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < 10; j++) {
            ASSERT_EQ(float(i * 10 + j), *ptr);
            ++ptr;
        }
        for (size_t j = 0; j < 10; j++) {
            ASSERT_EQ(float(i * 10 + j + 200), *ptr);
            ++ptr;
        }
    }
    slice0->fill_zero();
    Tensor tmp;
    tmp.copy_from(*slice0);
    float* tmp_ptr = static_cast<float*>(tmp.get_memory_ptr());
    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < 10; j++) {
            ASSERT_EQ(float(0), *tmp_ptr);
            ++tmp_ptr;
        }
    }
}

TEST(TestTensor, GetPtrOffset) {
    Layout layout{{20, 20}, 2};
    Tensor tensor(LiteDeviceType::LITE_CPU, layout);
    //! alloc memory
    auto ptr = static_cast<float*>(tensor.get_memory_ptr());

    auto ptr_offset = tensor.get_memory_ptr({10, 10});
    ASSERT_EQ(ptr_offset, ptr + 10 * 20 + 10);

    auto slice0 = tensor.slice({0, 0}, {20, 10});
    auto slice1 = tensor.slice({0, 10}, {20, 20});

    ASSERT_FALSE(slice0->is_continue_memory());
    ASSERT_FALSE(slice1->is_continue_memory());

    auto ptr_offset_slice0 = slice0->get_memory_ptr({6, 5});
    auto ptr_offset_slice1 = slice1->get_memory_ptr({2, 5});

    ASSERT_EQ(ptr_offset_slice0, ptr + 6 * 20 + 5);
    ASSERT_EQ(ptr_offset_slice1, ptr + 2 * 20 + 10 + 5);
}

TEST(TestTensor, Concat) {
    Layout layout{{5, 5, 5}, 3};
    std::vector<Tensor> tensors;
    for (int i = 0; i < 4; i++) {
        Tensor tensor(LiteDeviceType::LITE_CPU, layout);
        auto ptr = static_cast<float*>(tensor.get_memory_ptr());
        for (int n = 0; n < 5 * 5 * 5; n++) {
            ptr[n] = i;
        }
        tensors.push_back(tensor);
    }
    auto check = [&](int dim) {
        auto new_tensor = TensorUtils::concat(tensors, dim);
        auto ptr = static_cast<float*>(new_tensor->get_memory_ptr());
        size_t stride = std::pow(5, (3 - dim));
        for (int i = 0; i < 4; i++) {
            for (size_t j = 0; j < stride; j++) {
                ASSERT_EQ(ptr[i * stride + j], i);
            }
        }
    };
    check(0);
    check(1);
    check(2);
}

#if LITE_WITH_CUDA
TEST(TestTensor, BasicDevice) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor1(LiteDeviceType::LITE_CUDA, layout);
    Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
    //! mge tensor has created
    ASSERT_TRUE(TensorHelper::implement(&tensor1));
    ASSERT_TRUE(TensorHelper::implement(&tensor2));

    //! check member
    ASSERT_EQ(tensor1.get_device_type(), LiteDeviceType::LITE_CUDA);
    ASSERT_EQ(tensor2.get_device_type(), LiteDeviceType::LITE_CPU);
    ASSERT_EQ(tensor2.get_layout(), layout);
    //! check the real tensor
    ASSERT_EQ(tensor1.get_tensor_total_size_in_byte(), 1 * 3 * 224 * 224 * 4);
    ASSERT_EQ(tensor2.get_tensor_total_size_in_byte(), 1 * 3 * 224 * 224 * 4);

    ASSERT_TRUE(TensorHelper::implement(&tensor2)
                        ->cast_final_safe<TensorImplDft>()
                        .host_tensor());

    ASSERT_FALSE(TensorHelper::implement(&tensor2)
                         ->cast_final_safe<TensorImplDft>()
                         .dev_tensor());
    ASSERT_TRUE(TensorHelper::implement(&tensor1)
                        ->cast_final_safe<TensorImplDft>()
                        .dev_tensor());
    ASSERT_FALSE(TensorHelper::implement(&tensor1)
                         ->cast_final_safe<TensorImplDft>()
                         .host_tensor());
}

TEST(TestTensor, SetLayoutReAllocDevice) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor2(LiteDeviceType::LITE_CUDA, layout);
    auto old_ptr2 = tensor2.get_memory_ptr();

    //! layout set through
    Layout layout1{{1, 3, 100, 100}, 4, LiteDataType::LITE_INT8};
    tensor2.set_layout(layout1);
    ASSERT_EQ(tensor2.get_tensor_total_size_in_byte(), 1 * 3 * 100 * 100);
    auto layout2 = TensorHelper::implement(&tensor2)
                           ->cast_final_safe<TensorImplDft>()
                           .dev_tensor()
                           ->layout();
    ASSERT_EQ(to_lite_layout(layout2), layout1);

    auto new_ptr2 = tensor2.get_memory_ptr();

    ASSERT_EQ(old_ptr2, new_ptr2);
}

TEST(TestTensor, CrossCNCopyDevice) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor0;
    Tensor tensor1(LiteDeviceType::LITE_CPU);
    Tensor tensor2(LiteDeviceType::LITE_CPU, layout);
    Tensor tensor3(LiteDeviceType::LITE_CUDA, layout);

    tensor2.copy_from(tensor3);
    tensor3.copy_from(tensor2);

    auto old_ptr2 = tensor2.get_memory_ptr();
    auto old_ptr3 = tensor3.get_memory_ptr();
    ASSERT_THROW(tensor3.copy_from(tensor1), std::exception);

    tensor1.copy_from(tensor3);
    tensor0.copy_from(tensor3);

    tensor2.copy_from(tensor3);
    tensor3.copy_from(tensor2);

    ASSERT_EQ(tensor2.get_memory_ptr(), old_ptr2);
    ASSERT_EQ(tensor3.get_memory_ptr(), old_ptr3);
}

TEST(TestTensor, PinnedHostMem) {
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor1(LiteDeviceType::LITE_CPU);
    bool is_pinned_host = true;
    Tensor tensor2(LiteDeviceType::LITE_CUDA, layout, is_pinned_host);
    Tensor tensor3(LiteDeviceType::LITE_CUDA, layout);
    tensor2.copy_from(tensor3);
    tensor3.copy_from(tensor2);

    ASSERT_EQ(tensor2.is_pinned_host(), true);
    ASSERT_EQ(tensor3.is_pinned_host(), false);

    auto old_ptr2 = tensor2.get_memory_ptr();
    auto old_ptr3 = tensor3.get_memory_ptr();

    //! test source tenor is empty
    ASSERT_THROW(tensor2.copy_from(tensor1), std::exception);
    tensor1.copy_from(tensor2);
    tensor2.copy_from(tensor3);
    tensor3.copy_from(tensor2);

    ASSERT_EQ(tensor2.get_memory_ptr(), old_ptr2);
    ASSERT_EQ(tensor3.get_memory_ptr(), old_ptr3);
}

TEST(TestTensor, DeviceId) {
    if (get_device_count(LITE_CUDA) <= 1)
        return;
    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor2(0, LiteDeviceType::LITE_CUDA, layout);
    Tensor tensor3(1, LiteDeviceType::LITE_CUDA, layout);

    tensor2.copy_from(tensor3);
    tensor3.copy_from(tensor2);

    Tensor tensor1;
    tensor1.copy_from(tensor2);
    tensor1.copy_from(tensor3);
}

TEST(TestTensor, SliceDevice) {
    Layout layout{{20, 20}, 2};
    Tensor host_tensor0;
    Tensor dev_tensor0(LiteDeviceType::LITE_CUDA, layout);
    host_tensor0.copy_from(dev_tensor0);
    auto ptr = host_tensor0.get_memory_ptr();

    for (int i = 0; i < 20 * 20; i++) {
        *(static_cast<float*>(ptr) + i) = i;
    }
    dev_tensor0.copy_from(host_tensor0);

    auto check = [&](size_t start, size_t end, size_t step) {
        Tensor host_tensor;
        host_tensor.copy_from(
                *dev_tensor0.slice({start, start}, {end, end}, {step, step}));
        float* new_ptr = static_cast<float*>(host_tensor.get_memory_ptr());
        for (size_t i = start; i < end; i += step) {
            for (size_t j = start; j < end; j += step) {
                ASSERT_EQ(float(i * 20 + j), *new_ptr);
                ++new_ptr;
            }
        }
    };
    check(5, 10, 1);
    check(5, 11, 2);
    check(2, 18, 4);
}

TEST(TestTensor, MemSetDevice) {
    Layout layout{{20, 20}, 2, LiteDataType::LITE_INT8};
    Tensor host_tensor0(LiteDeviceType::LITE_CPU, layout);
    Tensor dev_tensor0(LiteDeviceType::LITE_CUDA, layout);
    auto check = [&](uint8_t val, const Tensor& tensor) {
        auto ptr = static_cast<uint8_t*>(tensor.get_memory_ptr());
        for (int i = 0; i < 20 * 20; i++) {
            ASSERT_EQ(val, *(ptr + i));
        }
    };
    host_tensor0.fill_zero();
    check(0, host_tensor0);

    Tensor host_tensor1;
    dev_tensor0.fill_zero();
    host_tensor1.copy_from(dev_tensor0);
    check(0, host_tensor1);
}

TEST(TestTensor, DeviceSliceCopy) {
    Layout layout{{20, 20}, 2};
    Tensor tensor(LiteDeviceType::LITE_CUDA, layout);
    //! alloc memory
    tensor.get_memory_ptr();

    Layout layout_slice{{20, 10}, 2};
    Tensor tensor0(LiteDeviceType::LITE_CPU, layout_slice);
    auto ptr0 = tensor0.get_memory_ptr();
    for (int i = 0; i < 10 * 20; i++) {
        *(static_cast<float*>(ptr0) + i) = i;
    }
    Tensor tensor1(LiteDeviceType::LITE_CPU, layout_slice);
    auto ptr1 = tensor1.get_memory_ptr();
    for (int i = 0; i < 10 * 20; i++) {
        *(static_cast<float*>(ptr1) + i) = i + 200;
    }

    auto slice0 = tensor.slice({0, 0}, {20, 10});
    auto slice1 = tensor.slice({0, 10}, {20, 20});

    slice0->copy_from(tensor0);
    slice1->copy_from(tensor1);

    ASSERT_FALSE(slice0->is_continue_memory());
    ASSERT_FALSE(slice1->is_continue_memory());

    Tensor host_tensor;
    host_tensor.copy_from(tensor);
    auto ptr = static_cast<float*>(host_tensor.get_memory_ptr());

    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < 10; j++) {
            ASSERT_EQ(float(i * 10 + j), *ptr);
            ++ptr;
        }
        for (size_t j = 0; j < 10; j++) {
            ASSERT_EQ(float(i * 10 + j + 200), *ptr);
            ++ptr;
        }
    }
    slice0->fill_zero();
    Tensor tmp;
    tmp.copy_from(*slice0);
    float* tmp_ptr = static_cast<float*>(tmp.get_memory_ptr());
    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < 10; j++) {
            ASSERT_EQ(float(0), *tmp_ptr);
            ++tmp_ptr;
        }
    }
}

TEST(TestTensor, ConcatDevice) {
    Layout layout{{5, 5, 5}, 3};
    std::vector<Tensor> tensors;
    for (int i = 0; i < 4; i++) {
        Tensor tensor(LiteDeviceType::LITE_CPU, layout);
        auto ptr = static_cast<float*>(tensor.get_memory_ptr());
        for (int n = 0; n < 5 * 5 * 5; n++) {
            ptr[n] = i;
        }
        tensors.push_back(tensor);
    }
    auto check = [&](int dim) {
        auto new_tensor =
                TensorUtils::concat(tensors, dim, LiteDeviceType::LITE_CUDA, 0);

        Tensor tensor(LiteDeviceType::LITE_CPU);
        tensor.copy_from(*new_tensor);
        auto ptr = static_cast<float*>(tensor.get_memory_ptr());
        size_t stride = std::pow(5, (3 - dim));
        for (int i = 0; i < 4; i++) {
            for (size_t j = 0; j < stride; j++) {
                ASSERT_EQ(ptr[i * stride + j], i);
            }
        }
        ASSERT_EQ(new_tensor->get_device_type(), LiteDeviceType::LITE_CUDA);
        ASSERT_EQ(new_tensor->get_device_id(), 0);
    };
    check(0);
    check(1);
    check(2);
}

TEST(TestTensor, CudaOutputDevice) {
    Layout layout{{1, 4}, 2};
    bool is_pinned_host = true;
    Tensor tensor(LiteDeviceType::LITE_CUDA, layout, is_pinned_host);
    // If is_pinned_host is true, when calling update_from_implement(), the device type
    // should always be updated with
    // get_device_from_locator(m_host_tensor->comp_node().locator()).
    tensor.update_from_implement();
    ASSERT_EQ(tensor.get_device_type(), LiteDeviceType::LITE_CUDA);
}
#endif
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
