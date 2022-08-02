#include "../src/misc.h"

#if LITE_BUILD_WITH_MGE
#include "../src/common.h"
#include "../src/mge/network_impl.h"

#include "../lite-c/src/common.h"
#include "lite-c/global_c.h"
#include "lite-c/network_c.h"
#include "lite-c/tensor_c.h"

#include "./test_common.h"
#include "megbrain/tensor.h"

#include <string.h>
#include <chrono>
#include <memory>
#include <random>
#include <unordered_map>

namespace {

int affinity_set = false;
int single_thread_affinity(int) {
    affinity_set = true;
    return 0;
}

std::atomic_size_t m_nr_left{0};
std::atomic_size_t m_nr_allocated{0};

void* allocate(LiteDeviceType device, int, size_t size, size_t align) {
    LITE_ASSERT(device == LiteDeviceType::LITE_CPU);
    m_nr_left++;
    m_nr_allocated++;
#ifdef WIN32
    return _aligned_malloc(size, align);
#elif defined(__ANDROID__) || defined(ANDROID)
    return memalign(align, size);
#else
    void* ptr = nullptr;
    auto err = posix_memalign(&ptr, align, size);
    mgb_assert(!err, "failed to malloc %zu bytes with align %zu", size, align);
    return ptr;
#endif
}

void free(LiteDeviceType device, int, void* ptr) {
    m_nr_left--;
    LITE_ASSERT(device == LiteDeviceType::LITE_CPU);
#ifdef WIN32
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
};

#define NUMBER_THREDS (4)
std::vector<std::thread::id> thread_ids(NUMBER_THREDS);
int multi_thread_affinity(int id) {
    thread_ids[id] = std::this_thread::get_id();
    return 0;
};

volatile bool finished = false;
int async_callback() {
    finished = true;
    return 0;
}

volatile bool finished_with_data = false;
int async_callback_with_data(void* user_data) {
    if (user_data != NULL) {
        std::cout << "async_callback user_data addr=" << std::hex << user_data
                  << std::endl;
    }
    finished_with_data = true;
    return 0;
}

volatile bool start_checked = false;
int start_callback(const LiteIO* inputs, const LiteTensor* input_tensors, size_t size) {
    start_checked = true;
    auto check_func = [&]() {
        ASSERT_EQ(size, 1);
        ASSERT_EQ(std::string(inputs->name), "data");
        LiteLayout layout;
        LITE_get_tensor_layout(*input_tensors, &layout);
        ASSERT_EQ(layout.ndim, 4);
        ASSERT_EQ(layout.shapes[1], 3);
        ASSERT_EQ(layout.shapes[2], 224);
        ASSERT_EQ(layout.shapes[3], 224);
    };
    check_func();
    return 0;
}

volatile bool start_checked_with_data = false;
int start_callback_with_data(
        const LiteIO* inputs, const LiteTensor* input_tensors, size_t size,
        void* user_data) {
    start_checked_with_data = true;
    auto check_func = [&]() {
        if (user_data != NULL) {
            std::cout << "start_callback user_data addr=" << std::hex << user_data
                      << std::endl;
        }
        ASSERT_EQ(size, 1);
        ASSERT_EQ(std::string(inputs->name), "data");
        LiteLayout layout;
        LITE_get_tensor_layout(*input_tensors, &layout);
        ASSERT_EQ(layout.ndim, 4);
        ASSERT_EQ(layout.shapes[1], 3);
        ASSERT_EQ(layout.shapes[2], 224);
        ASSERT_EQ(layout.shapes[3], 224);
    };
    check_func();
    return 0;
}

volatile bool finish_checked = false;
int finish_callback(
        const LiteIO* outputs, const LiteTensor* output_tensors, size_t size) {
    finish_checked = true;
    auto check_func = [&]() {
        ASSERT_EQ(size, 1);
        ASSERT_EQ(
                std::string(outputs->name),
                "TRUE_DIV(EXP[12065],reduce0[12067])[12077]");
        LiteLayout layout;
        LITE_get_tensor_layout(*output_tensors, &layout);
        ASSERT_EQ(layout.shapes[1], 1000);
    };
    check_func();
    return 0;
}

volatile bool finish_checked_with_data = false;
int finish_callback_with_data(
        const LiteIO* outputs, const LiteTensor* output_tensors, size_t size,
        void* user_data) {
    finish_checked_with_data = true;
    auto check_func = [&]() {
        if (user_data != NULL) {
            std::cout << "finish_callback user_data addr=" << std::hex << user_data
                      << std::endl;
        }
        ASSERT_EQ(size, 1);
        ASSERT_EQ(
                std::string(outputs->name),
                "TRUE_DIV(EXP[12065],reduce0[12067])[12077]");
        LiteLayout layout;
        LITE_get_tensor_layout(*output_tensors, &layout);
        ASSERT_EQ(layout.shapes[1], 1000);
    };
    check_func();
    return 0;
}

}  // namespace

#define LITE_CAPI_CHECK(_expr)                 \
    do {                                       \
        int _ret = (_expr);                    \
        if (_ret) {                            \
            LITE_THROW(LITE_get_last_error()); \
        }                                      \
    } while (0)

#define ForwardMgb                                                             \
    lite::Config config;                                                       \
    auto lite_tensor = lite::get_input_data("./input_data.npy");               \
    size_t data_length_in_byte = lite_tensor->get_tensor_total_size_in_byte(); \
    std::string model_path = "./shufflenet.mge";                               \
    auto result_mgb = mgb_lar(model_path, config, "data", lite_tensor)

#define MakeNetwork        \
    LiteNetwork c_network; \
    LITE_CAPI_CHECK(       \
            LITE_make_network(&c_network, *default_config(), *default_network_io()))

#define LoadNetwork \
    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, model_path.c_str()))

#define SetInput                                                                 \
    LiteTensor c_input_tensor, c_output_tensor;                                  \
    LITE_CAPI_CHECK(                                                             \
            LITE_get_io_tensor(c_network, "data", LITE_INPUT, &c_input_tensor)); \
    LITE_CAPI_CHECK(LITE_reset_tensor_memory(                                    \
            c_input_tensor, lite_tensor->get_memory_ptr(), data_length_in_byte))

#define ForwardNetwork                        \
    LITE_CAPI_CHECK(LITE_forward(c_network)); \
    LITE_CAPI_CHECK(LITE_wait(c_network))

#define GetOutput                                                      \
    const char* output_name;                                           \
    LITE_CAPI_CHECK(LITE_get_output_name(c_network, 0, &output_name)); \
    LITE_CAPI_CHECK(LITE_get_io_tensor(                                \
            c_network, output_name, LITE_OUTPUT, &c_output_tensor));   \
    void* output_ptr;                                                  \
    LITE_CAPI_CHECK(LITE_get_tensor_memory(c_output_tensor, &output_ptr))

#define CompareResult                                 \
    EXPECT_TRUE(lite::compare_memory<float>(          \
            output_ptr, result_mgb->get_memory_ptr(), \
            result_mgb->get_tensor_total_size_in_byte() / sizeof(float)))

TEST(TestCapiNetWork, BasicResetInput) {
    ForwardMgb;
    LiteNetwork c_network;
    LITE_CAPI_CHECK(LITE_make_default_network(&c_network));
    LoadNetwork;
    SetInput;
    ForwardNetwork;
    GetOutput;
    CompareResult;
    LITE_destroy_network(c_network);
}

TEST(TestCapiNetWork, GetAllName) {
    std::string model_path = "./shufflenet.mge";
    LiteNetwork c_network;
    LITE_CAPI_CHECK(LITE_make_default_network(&c_network));
    LoadNetwork;
    size_t input_size, output_size;
    LITE_get_all_input_name(c_network, &input_size, nullptr);
    LITE_get_all_output_name(c_network, &output_size, nullptr);

    std::vector<const char*> input_names(input_size);
    LITE_get_all_input_name(c_network, nullptr, input_names.data());
    ASSERT_EQ(input_names.size(), 1);
    ASSERT_TRUE(std::string(input_names[0]) == "data");

    std::vector<const char*> output_names(output_size);
    LITE_get_all_output_name(c_network, nullptr, output_names.data());
    ASSERT_TRUE(
            std::string(output_names[0]) ==
            "TRUE_DIV(EXP[12065],reduce0[12067])[12077]");
    ASSERT_EQ(output_names.size(), 1);
    LITE_destroy_network(c_network);
}

TEST(TestCapiNetWork, GetAllNameAhead) {
    std::string model_path = "./shufflenet.mge";
    LiteNetworkIO ios, ios_mem;
    LITE_CAPI_CHECK(LITE_get_model_io_info_by_path(
            model_path.c_str(), *default_config(), &ios));
    FILE* fin = fopen(model_path.c_str(), "rb");
    ASSERT_TRUE(fin);
    fseek(fin, 0, SEEK_END);
    size_t size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    void* ptr = malloc(size);
    std::shared_ptr<void> buf{ptr, ::free};
    auto nr = fread(buf.get(), 1, size, fin);
    LITE_ASSERT(nr == size);
    fclose(fin);

    LITE_CAPI_CHECK(
            LITE_get_model_io_info_by_memory(ptr, size, *default_config(), &ios_mem));

    ASSERT_EQ(ios.input_size, 1);
    ASSERT_EQ(ios.output_size, 1);
    ASSERT_EQ(ios_mem.input_size, 1);
    ASSERT_EQ(ios_mem.output_size, 1);

    ASSERT_TRUE(std::string(ios.inputs->name) == "data");
    ASSERT_TRUE(ios.inputs->config_layout.ndim == 4);
    ASSERT_TRUE(ios.inputs->config_layout.shapes[1] == 3);
    ASSERT_TRUE(ios.inputs->config_layout.shapes[2] == 224);
    ASSERT_TRUE(ios.inputs->config_layout.shapes[3] == 224);
    ASSERT_TRUE(
            std::string(ios.outputs->name) ==
            "TRUE_DIV(EXP[12065],reduce0[12067])[12077]");
    ASSERT_TRUE(ios.outputs->config_layout.ndim == 2);
    ASSERT_TRUE(ios.outputs->config_layout.shapes[0] == 1);
    ASSERT_TRUE(ios.outputs->config_layout.shapes[1] == 1000);

    ASSERT_TRUE(std::string(ios_mem.inputs->name) == "data");
    ASSERT_TRUE(ios_mem.inputs->config_layout.ndim == 4);
    ASSERT_TRUE(ios_mem.inputs->config_layout.shapes[1] == 3);
    ASSERT_TRUE(ios_mem.inputs->config_layout.shapes[2] == 224);
    ASSERT_TRUE(ios_mem.inputs->config_layout.shapes[3] == 224);
    ASSERT_TRUE(
            std::string(ios_mem.outputs->name) ==
            "TRUE_DIV(EXP[12065],reduce0[12067])[12077]");
    ASSERT_TRUE(ios_mem.outputs->config_layout.ndim == 2);
    ASSERT_TRUE(ios_mem.outputs->config_layout.shapes[0] == 1);
    ASSERT_TRUE(ios_mem.outputs->config_layout.shapes[1] == 1000);
}

#if LITE_BUILD_WITH_RKNPU

static int GetTop(
        float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount,
        uint32_t topNum) {
    uint32_t i, j;

#define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM)
        return 0;

    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++) {
        for (i = 0; i < outputCount; i++) {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) ||
                (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
                (i == *(pMaxClass + 4))) {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb + j)) {
                *(pfMaxProb + j) = pfProb[i];
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

TEST(TestCapiNetWork, rknntest_set_info) {
#define SET_INFO_SIZE      2
#define TENSOR_TYPE_UINT8  3
#define TENSOR_FORMAT_NHWC 1
    LiteConfig config;
    config.backend = LiteBackend::LITE_RK_NPU;
    config.device_type = LiteDeviceType::LITE_NPU;
    config.bare_model_cryption_name = nullptr;
    auto lite_tensor = lite::get_input_data("./model/cat_224x224.npy");
    auto true_tensor = lite::get_input_data("./output_data.npy");
    auto rknn_model = "./model/mobilenet_v1.rknn";

    LiteNetwork c_network;
    LITE_CAPI_CHECK(LITE_make_network_config(&c_network, config));
    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, rknn_model));

    size_t input_size, output_size;
    LITE_get_all_input_name(c_network, &input_size, nullptr);
    LITE_get_all_output_name(c_network, &output_size, nullptr);

    std::vector<const char*> input_names(input_size);
    std::vector<const char*> output_names(output_size);
    LiteTensor c_input_tensor, c_output_tensor;

    LITE_get_all_input_name(c_network, nullptr, input_names.data());
    LITE_get_all_output_name(c_network, nullptr, output_names.data());
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, input_names[0], LITE_IO, &c_input_tensor));

    size_t input_length = 0;
    LITE_get_tensor_total_size_in_byte(c_input_tensor, &input_length);

    size_t data_length_in_byte = lite_tensor->get_tensor_total_size_in_byte();
    {
        LiteLayout input_layout;
        LITE_get_tensor_layout(c_input_tensor, &input_layout);
        ASSERT_TRUE(input_layout.data_type == LITE_INT8);
        std::vector<int> input_shape = {1, 224, 224, 3};
        for (size_t i = 0; i < input_layout.ndim; i++) {
            ASSERT_TRUE(input_layout.shapes[i] = input_shape[i]);
        }
    }

    {
        int size_attr = 0;
        LITE_CAPI_CHECK(LITE_get_tensor_attribute(
                c_input_tensor, nullptr, nullptr, &size_attr));
        ASSERT_TRUE(size_attr > 0);
        const char* keys[size_attr];
        void* values[size_attr];
        LITE_CAPI_CHECK(
                LITE_get_tensor_attribute(c_input_tensor, keys, values, &size_attr));
        ASSERT_TRUE(size_attr > 5);
        std::unordered_map<std::string, uint32_t> result_map = {
                {"zp", 0},       {"index", 0},       {"size_with_stride", 150528},
                {"stride", 224}, {"n_size", 150528}, {"n_elems", 150528},
                {"qnt_type", 2}, {"n_dims", 4},      {"type", 2},
                {"fmt", 1},      {"dims0", 1},       {"dims1", 224},
                {"dims2", 224},  {"dims3", 3},
        };
        for (int i = 0; i < size_attr; i++) {
            std::string key(keys[i]);
            if (key == "names") {
                ASSERT_TRUE(
                        std::string("input") ==
                        std::string(static_cast<const char*>(values[i])));
            } else if (key == "scale") {
                float scale = *static_cast<float*>(values[i]);
                ASSERT_TRUE(std::fabs(scale - 0.007812) < 0.00001);
            } else if (key == "fl" || key == "pass_through") {
                uint8_t val = *static_cast<uint8_t*>(values[i]);
                if (key == "fl") {
                    ASSERT_TRUE(val == 0);
                } else {
                    ASSERT_TRUE(val == 1);
                }
            } else {
                uint32_t val = *static_cast<uint32_t*>(values[i]);
                ASSERT_TRUE(result_map[std::string(keys[i])] == val);
            }
        }
    }
    const char* keys[] = {"type", "fmt"};
    int info_size = SET_INFO_SIZE;
    int type = TENSOR_TYPE_UINT8;
    int fmt = TENSOR_FORMAT_NHWC;
    void* values[] = {static_cast<void*>(&type), static_cast<void*>(&fmt)};
    LITE_CAPI_CHECK(
            LITE_set_tensor_information(c_input_tensor, keys, values, info_size));
    ASSERT_TRUE(
            std::string(output_names[0]) ==
            std::string("MobilenetV1/Predictions/Reshape_1"));
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, output_names[0], LITE_IO, &c_output_tensor));

    LITE_CAPI_CHECK(LITE_reset_tensor_memory(
            c_input_tensor, lite_tensor->get_memory_ptr(), data_length_in_byte));

    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, output_names[0], LITE_IO, &c_output_tensor));
    // LiteLayout tmp_output_layout;
    // LITE_get_tensor_layout(c_output_tensor, &tmp_output_layout);
    // tmp_output_layout.data_type = LiteDataType::LITE_FLOAT;

    // LITE_set_tensor_layout(c_output_tensor, tmp_output_layout);
    {
        const char* keys[] = {"want_float"};
        uint8_t want_float = 1;
        void* values[] = {static_cast<void*>(&want_float)};
        LITE_CAPI_CHECK(LITE_set_tensor_information(c_output_tensor, keys, values, 1));
    }

    LITE_CAPI_CHECK(LITE_forward(c_network));
    LITE_CAPI_CHECK(LITE_wait(c_network));

    ASSERT_TRUE(std::string(output_names[0]) == "MobilenetV1/Predictions/Reshape_1");
    ASSERT_EQ(output_names.size(), 1);
    {
        LiteLayout output_layout;
        LITE_get_tensor_layout(c_output_tensor, &output_layout);
        ASSERT_TRUE(output_layout.data_type == LITE_FLOAT);
        int size_attr = 0;

        LITE_CAPI_CHECK(LITE_get_tensor_attribute(
                c_output_tensor, nullptr, nullptr, &size_attr));
        ASSERT_TRUE(size_attr > 0);
        const char* keys[size_attr];
        void* values[size_attr];
        LITE_CAPI_CHECK(
                LITE_get_tensor_attribute(c_output_tensor, keys, values, &size_attr));
        ASSERT_TRUE(size_attr > 5);
        std::unordered_map<std::string, uint32_t> result_map = {
                {"zp", 0},       {"index", 0},     {"size_with_stride", 2002},
                {"stride", 0},   {"n_size", 2002}, {"n_elems", 1001},
                {"qnt_type", 2}, {"n_dims", 2},    {"type", 0},
                {"fmt", 2},      {"dims0", 1},     {"dims1", 1001},
        };
        for (int i = 0; i < size_attr; i++) {
            std::string key(keys[i]);
            if (key == "names") {
                ASSERT_TRUE(
                        "MobilenetV1/Predictions/Reshape_1" ==
                        std::string(static_cast<const char*>(values[i])));

            } else if (key == "scale") {
                float scale = *static_cast<float*>(values[i]);
                ASSERT_TRUE(std::fabs(scale - 1.0) < 0.00001);
            } else if (key == "fl" || key == "pass_through") {
                uint8_t val = *static_cast<uint8_t*>(values[i]);
                ASSERT_TRUE(val == 0);
            } else {
                uint32_t val = *static_cast<uint32_t*>(values[i]);
                ASSERT_TRUE(result_map[std::string(keys[i])] == val);
            }
        }
    }
    {
        uint32_t MaxClass[5];
        float fMaxProb[5];
        void* output_ptr;
        LITE_get_tensor_memory(c_output_tensor, &output_ptr);
        float* buffer = (float*)output_ptr;
        uint32_t sz = true_tensor->get_tensor_total_size_in_byte() / sizeof(float);

        GetTop(buffer, fMaxProb, MaxClass, sz, 5);

        std::vector<uint32_t> result_class = {
                286, 464, 282, 357, 285,
        };
        std::vector<float> result_prob = {
                0.407227, 0.365723, 0.090454, 0.018051, 0.013069,
        };

        for (int i = 0; i < 5; i++) {
            ASSERT_TRUE(result_class[i] == MaxClass[i]);
            ASSERT_TRUE(std::fabs(result_prob[i] - fMaxProb[i]) < 0.0001);
        }
    }

    {
        float* true_data = static_cast<float*>(true_tensor->get_memory_ptr());
        void* output_ptr;
        LITE_get_tensor_memory(c_output_tensor, &output_ptr);
        float* data1 = static_cast<float*>(output_ptr);
        size_t length = true_tensor->get_tensor_total_size_in_byte() / sizeof(float);
        for (size_t i = 0; i < length; i++) {
            ASSERT_LT(std::abs(data1[i] - true_data[i]), 1e-3);
        }
    }
    LITE_destroy_network(c_network);
#undef SET_INFO_SIZE
#undef TENSOR_FORMAT_NHWC
#undef TENSOR_TYPE_UINT8
}

TEST(TestCapiNetWork, rknntest_set_info_two_input) {
#define SET_INFO_SIZE      2
#define TENSOR_TYPE_UINT8  3
#define TENSOR_FORMAT_NHWC 1
    LiteConfig config;
    config.backend = LiteBackend::LITE_RK_NPU;
    config.device_type = LiteDeviceType::LITE_NPU;
    config.bare_model_cryption_name = nullptr;
    auto lite_tensor = lite::get_input_data("./model/cat_224x224.npy");
    auto lite_tensor_dog = lite::get_input_data("./model/dog_224x224.npy");
    auto true_tensor = lite::get_input_data("./output_data.npy");
    auto rknn_model = "./model/mobilenet_v1.rknn";

    LiteNetwork c_network;
    LITE_CAPI_CHECK(LITE_make_network_config(&c_network, config));
    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, rknn_model));

    size_t input_size, output_size;
    LITE_get_all_input_name(c_network, &input_size, nullptr);
    LITE_get_all_output_name(c_network, &output_size, nullptr);

    std::vector<const char*> input_names(input_size);
    std::vector<const char*> output_names(output_size);
    LiteTensor c_input_tensor, c_output_tensor;

    LITE_get_all_input_name(c_network, nullptr, input_names.data());
    LITE_get_all_output_name(c_network, nullptr, output_names.data());
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, input_names[0], LITE_IO, &c_input_tensor));

    size_t input_length = 0;
    LITE_get_tensor_total_size_in_byte(c_input_tensor, &input_length);

    size_t data_length_in_byte = lite_tensor->get_tensor_total_size_in_byte();
    {
        LiteLayout input_layout;
        LITE_get_tensor_layout(c_input_tensor, &input_layout);
        ASSERT_TRUE(input_layout.data_type == LITE_INT8);
        std::vector<int> input_shape = {1, 224, 224, 3};
        for (size_t i = 0; i < input_layout.ndim; i++) {
            ASSERT_TRUE(input_layout.shapes[i] = input_shape[i]);
        }
    }

    const char* keys[] = {"type", "fmt"};
    int info_size = SET_INFO_SIZE;
    int type = TENSOR_TYPE_UINT8;
    int fmt = TENSOR_FORMAT_NHWC;
    void* values[] = {static_cast<void*>(&type), static_cast<void*>(&fmt)};
    LITE_CAPI_CHECK(
            LITE_set_tensor_information(c_input_tensor, keys, values, info_size));
    ASSERT_TRUE(
            std::string(output_names[0]) ==
            std::string("MobilenetV1/Predictions/Reshape_1"));
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, output_names[0], LITE_IO, &c_output_tensor));

    LITE_CAPI_CHECK(LITE_reset_tensor_memory(
            c_input_tensor, lite_tensor->get_memory_ptr(), data_length_in_byte));

    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, output_names[0], LITE_IO, &c_output_tensor));
    {
        const char* keys[] = {"want_float"};
        uint8_t want_float = 1;
        void* values[] = {static_cast<void*>(&want_float)};
        LITE_CAPI_CHECK(LITE_set_tensor_information(c_output_tensor, keys, values, 1));
    }

    LITE_CAPI_CHECK(LITE_forward(c_network));
    LITE_CAPI_CHECK(LITE_wait(c_network));

    ASSERT_TRUE(std::string(output_names[0]) == "MobilenetV1/Predictions/Reshape_1");
    ASSERT_EQ(output_names.size(), 1);
    {
        uint32_t MaxClass[5];
        float fMaxProb[5];
        void* output_ptr;
        LITE_get_tensor_memory(c_output_tensor, &output_ptr);
        float* buffer = (float*)output_ptr;
        uint32_t sz = true_tensor->get_tensor_total_size_in_byte() / sizeof(float);

        GetTop(buffer, fMaxProb, MaxClass, sz, 5);

        std::vector<uint32_t> result_class = {
                286, 464, 282, 357, 285,
        };
        std::vector<float> result_prob = {
                0.407227, 0.365723, 0.090454, 0.018051, 0.013069,
        };

        for (int i = 0; i < 5; i++) {
            ASSERT_TRUE(result_class[i] == MaxClass[i]);
            ASSERT_TRUE(std::fabs(result_prob[i] - fMaxProb[i]) < 0.0001);
        }
    }

    {
        float* true_data = static_cast<float*>(true_tensor->get_memory_ptr());
        void* output_ptr;
        LITE_get_tensor_memory(c_output_tensor, &output_ptr);
        float* data1 = static_cast<float*>(output_ptr);
        size_t length = true_tensor->get_tensor_total_size_in_byte() / sizeof(float);
        for (size_t i = 0; i < length; i++) {
            ASSERT_LT(std::abs(data1[i] - true_data[i]), 1e-3);
        }
    }

    LITE_CAPI_CHECK(LITE_reset_tensor_memory(
            c_input_tensor, lite_tensor_dog->get_memory_ptr(), data_length_in_byte));
    LITE_CAPI_CHECK(LITE_forward(c_network));
    LITE_CAPI_CHECK(LITE_wait(c_network));
    ASSERT_TRUE(std::string(output_names[0]) == "MobilenetV1/Predictions/Reshape_1");
    ASSERT_EQ(output_names.size(), 1);
    {
        uint32_t MaxClass[5];
        float fMaxProb[5];
        void* output_ptr;
        LITE_get_tensor_memory(c_output_tensor, &output_ptr);
        float* buffer = (float*)output_ptr;
        uint32_t sz = true_tensor->get_tensor_total_size_in_byte() / sizeof(float);

        GetTop(buffer, fMaxProb, MaxClass, sz, 5);

        std::vector<float> result_prob = {
                0.407227, 0.365723, 0.090454, 0.018051, 0.013069,
        };

        for (int i = 0; i < 5; i++) {
            ASSERT_FALSE(std::fabs(result_prob[i] - fMaxProb[i]) < 0.0001);
        }
    }

    LITE_destroy_network(c_network);
#undef SET_INFO_SIZE
#undef TENSOR_FORMAT_NHWC
#undef TENSOR_TYPE_UINT8
}
#endif

TEST(TestCapiNetWork, BasicResetOutput) {
    ForwardMgb;
    LiteNetwork c_network;
    LITE_CAPI_CHECK(LITE_make_default_network(&c_network));
    LoadNetwork;
    SetInput;
    LiteLayout output_layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT};
    std::shared_ptr<float> ptr(new float[1000], [](float* ptr) { delete[] ptr; });
    const char* output_name;
    LITE_CAPI_CHECK(LITE_get_output_name(c_network, 0, &output_name));
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network, output_name, LITE_IO, &c_output_tensor));
    LITE_CAPI_CHECK(LITE_reset_tensor(c_output_tensor, output_layout, ptr.get()));

    ForwardNetwork;

    EXPECT_TRUE(lite::compare_memory<float>(
            ptr.get(), result_mgb->get_memory_ptr(),
            result_mgb->get_tensor_total_size_in_byte() / sizeof(float)));
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, BasicInplaceAndSingleThreadAffinity) {
    ForwardMgb;
    MakeNetwork;
    //! config the network with cpu inplace mode
    LITE_CAPI_CHECK(LITE_set_cpu_inplace_mode(c_network));
    LoadNetwork;
    //! set single thread affinith callback
    LITE_CAPI_CHECK(
            LITE_set_runtime_thread_affinity(c_network, single_thread_affinity));
    SetInput;
    ForwardNetwork;
    ASSERT_EQ(affinity_set, true);
    affinity_set = false;
    GetOutput;
    CompareResult;
    LITE_destroy_network(c_network);
}

TEST(TestCapiNetWork, UserAllocator) {
    ForwardMgb;
    MakeNetwork;
    LITE_CAPI_CHECK(LITE_set_memory_allocator(c_network, allocate, free));
    LoadNetwork;
    SetInput;
    ForwardNetwork;

    ASSERT_GE(m_nr_allocated, 1);
    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
    ASSERT_EQ(m_nr_left, 0);
}

TEST(TestCapiNetWork, BasicMultiThread) {
    ForwardMgb;
    MakeNetwork;
    LITE_CAPI_CHECK(LITE_set_cpu_threads_number(c_network, NUMBER_THREDS));
    LoadNetwork;
    LITE_CAPI_CHECK(LITE_set_runtime_thread_affinity(c_network, multi_thread_affinity));
    SetInput;
    ForwardNetwork;
    for (size_t i = 0; i < NUMBER_THREDS; i++) {
        for (size_t j = i + 1; j < NUMBER_THREDS; j++) {
            ASSERT_NE(thread_ids[i], thread_ids[j]);
        }
    }
    for (size_t i = 0; i < NUMBER_THREDS; i++) {
        thread_ids[i] = std::thread::id();
    }
    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, DeviceIO) {
    ForwardMgb;
    LiteNetwork c_network;
    LiteIO input_io = default_io;
    input_io.is_host = true;
    input_io.name = "data";
    LiteNetworkIO network_io = *default_network_io();
    network_io.inputs = &input_io;
    network_io.input_size = 1;
    LITE_CAPI_CHECK(LITE_make_network(&c_network, *default_config(), network_io));
    LoadNetwork;
    SetInput;
    ForwardNetwork;
    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, StartCallBack) {
    ForwardMgb;
    MakeNetwork;
    LoadNetwork;
    LITE_CAPI_CHECK(LITE_set_start_callback(c_network, start_callback));
    SetInput;
    ForwardNetwork;
    GetOutput;
    CompareResult;
    ASSERT_TRUE(start_checked);
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, StartCallBackWithData) {
    ForwardMgb;
    MakeNetwork;
    LoadNetwork;
    size_t user_data = 1;
    LITE_CAPI_CHECK(LITE_set_start_callback_with_userdata(
            c_network, start_callback_with_data, &user_data));
    SetInput;
    ForwardNetwork;
    GetOutput;
    CompareResult;
    ASSERT_TRUE(start_checked_with_data);
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, FinishCallBack) {
    ForwardMgb;
    MakeNetwork;
    LoadNetwork;
    LITE_CAPI_CHECK(LITE_set_finish_callback(c_network, finish_callback));
    SetInput;
    ForwardNetwork;
    GetOutput;
    CompareResult;
    ASSERT_TRUE(finish_checked);
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, FinishCallBackWtihData) {
    ForwardMgb;
    MakeNetwork;
    LoadNetwork;
    size_t user_data = 1;
    LITE_CAPI_CHECK(LITE_set_finish_callback_with_userdata(
            c_network, finish_callback_with_data, &user_data));
    SetInput;
    ForwardNetwork;
    GetOutput;
    CompareResult;
    ASSERT_TRUE(finish_checked_with_data);
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, BasicCryptAes) {
    ForwardMgb;

    LiteConfig c_config = *default_config();
    c_config.bare_model_cryption_name = "AES_default";
    LiteNetwork c_network;
    LITE_CAPI_CHECK(LITE_make_network(&c_network, c_config, *default_network_io()));
    std::string model_crypt_path = "./shufflenet_crypt_aes.mge";

    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, model_crypt_path.c_str()));

    SetInput;
    ForwardNetwork;
    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, PackedCryptRc4) {
    ForwardMgb;
    MakeNetwork;

    std::string model_crypt_path = "./test_packed_model_rc4.lite";
    LITE_CAPI_CHECK(LITE_load_model_from_path(c_network, model_crypt_path.c_str()));

    SetInput;
    ForwardNetwork;
    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, AsyncExec) {
    finished = false;
    ForwardMgb;
    LiteNetwork c_network;
    LiteConfig c_config = *default_config();
    c_config.options.var_sanity_check_first_run = false;
    LITE_CAPI_CHECK(LITE_make_network(&c_network, c_config, *default_network_io()));
    LITE_CAPI_CHECK(LITE_set_async_callback(c_network, async_callback));
    LoadNetwork;
    SetInput;

    LITE_forward(c_network);
    size_t count = 0;
    while (finished == false) {
        count++;
    }
    ASSERT_GT(count, 0);
    finished = false;

    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, AsyncExecWithData) {
    finished = false;
    ForwardMgb;
    LiteNetwork c_network;
    LiteConfig c_config = *default_config();
    c_config.options.var_sanity_check_first_run = false;
    LITE_CAPI_CHECK(LITE_make_network(&c_network, c_config, *default_network_io()));
    size_t user_data = 1;
    LITE_CAPI_CHECK(LITE_set_async_callback_with_userdata(
            c_network, async_callback_with_data, &user_data));
    LoadNetwork;
    SetInput;

    LITE_forward(c_network);
    size_t count = 0;
    while (finished_with_data == false) {
        count++;
    }
    ASSERT_GT(count, 0);
    finished_with_data = false;

    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, OutputShapeOnly) {
    ForwardMgb;
    LiteNetwork c_network;
    LiteNetworkIO c_network_io = *default_network_io();
    LiteIO io_output = default_io;
    io_output.io_type = LiteIOType::LITE_IO_SHAPE;
    io_output.name = "TRUE_DIV(EXP[12065],reduce0[12067])[12077]";
    c_network_io.outputs = &io_output;
    c_network_io.output_size = 1;
    LITE_CAPI_CHECK(LITE_make_network(&c_network, *default_config(), c_network_io));
    LoadNetwork;
    SetInput;
    ForwardNetwork;
    GetOutput;
    size_t length = 0;
    LITE_CAPI_CHECK(LITE_get_tensor_total_size_in_byte(c_output_tensor, &length));
    ASSERT_EQ(length / sizeof(float), 1000);
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, ProfileIOdump) {
    ForwardMgb;
    MakeNetwork;
    LITE_CAPI_CHECK(LITE_enable_profile_performance(c_network, "./profile.json"));
    LoadNetwork;
    SetInput;
    ForwardNetwork;
    ASSERT_TRUE(fopen("./profile.json", "r"));

    LITE_CAPI_CHECK(LITE_enable_io_txt_dump(c_network, "./io_txt_dump.txt"));
    ForwardNetwork;
    ASSERT_TRUE(fopen("./io_txt_dump.txt", "r"));

    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, GlabalLayoutTransform) {
    ForwardMgb;
    MakeNetwork;
    LITE_CAPI_CHECK(LITE_enable_global_layout_transform(c_network));
    LoadNetwork;
    LITE_CAPI_CHECK(LITE_dump_layout_transform_model(
            c_network, "./shufflenet_after_trans.mge"));
    SetInput;
    ForwardNetwork;
    ASSERT_TRUE(fopen("./shufflenet_after_trans.mge", "r"));
    GetOutput;
    CompareResult;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, GetDeviceType) {
    lite::Config config;
    auto lite_tensor = lite::get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    MakeNetwork;
    LoadNetwork;
    LiteDeviceType devicetype;
    LITE_CAPI_CHECK(LITE_get_device_type(c_network, &devicetype));
    ASSERT_TRUE(devicetype == LiteDeviceType::LITE_CPU);
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, GetModelExtraInfo) {
    lite::Config config;
    std::string model_path = "./track_640_320_pack_model_rc4_with_info.lite";
    MakeNetwork;
    LITE_load_model_from_path(c_network, model_path.c_str());
    const char* info = nullptr;
    int info_size = 0;
    LITE_CAPI_CHECK(LITE_get_model_extra_info(c_network, &info, &info_size));
    ASSERT_TRUE(info_size > 0);
    printf("info %s \n", info);
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, TestWorkSpaceLimit) {
    lite::Config config;
    auto lite_tensor = lite::get_input_data("./input_data.npy");
    size_t data_length_in_byte = lite_tensor->get_tensor_total_size_in_byte();
    std::string model_path = "./shufflenet.mge";
    MakeNetwork;
    LoadNetwork;
    printf("go to config workspace limit\n");
    LITE_CAPI_CHECK(LITE_set_network_algo_workspace_limit(c_network, 1000));
    SetInput;
    ForwardNetwork;

    GetOutput;
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

TEST(TestCapiNetWork, TestShareWeights) {
    ForwardMgb;
    MakeNetwork;
    LoadNetwork;
    SetInput;
    ForwardNetwork;

    GetOutput;
    CompareResult;

    LiteNetwork c_network2;
    LITE_CAPI_CHECK(
            LITE_make_network(&c_network2, *default_config(), *default_network_io()));
    LITE_CAPI_CHECK(LITE_set_cpu_inplace_mode(c_network2));
    LITE_CAPI_CHECK(LITE_shared_weight_with_network(c_network2, c_network));
    int is_cpu_inplace_mode = false;
    LITE_CAPI_CHECK(LITE_is_cpu_inplace_mode(c_network2, &is_cpu_inplace_mode));
    ASSERT_EQ(is_cpu_inplace_mode, true);

    LiteTensor c_input_tensor2, c_output_tensor2;
    LITE_CAPI_CHECK(LITE_get_io_tensor(c_network2, "data", LITE_IO, &c_input_tensor2));
    LITE_CAPI_CHECK(LITE_reset_tensor_memory(
            c_input_tensor2, lite_tensor->get_memory_ptr(),
            lite_tensor->get_tensor_total_size_in_byte()));
    LITE_CAPI_CHECK(LITE_forward(c_network2));
    LITE_CAPI_CHECK(LITE_wait(c_network2));
    LITE_CAPI_CHECK(
            LITE_get_io_tensor(c_network2, output_name, LITE_IO, &c_output_tensor2));
    void* output_ptr2;
    LITE_CAPI_CHECK(LITE_get_tensor_memory(c_output_tensor2, &output_ptr2));

    EXPECT_TRUE(lite::compare_memory<float>(
            output_ptr2, result_mgb->get_memory_ptr(),
            result_mgb->get_tensor_total_size_in_byte() / sizeof(float)));

    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
    LITE_CAPI_CHECK(LITE_destroy_network(c_network2));
}

TEST(TestCapiNetWork, GlobalHolder) {
    std::string model_path = "./shufflenet.mge";
    LiteNetwork c_network;
    LITE_CAPI_CHECK(
            LITE_make_network(&c_network, *default_config(), *default_network_io()));
    auto destroy_network = c_network;
    LITE_CAPI_CHECK(
            LITE_make_network(&c_network, *default_config(), *default_network_io()));
    //! make sure destroy_network is destroyed by LITE_make_network
    LITE_destroy_network(destroy_network);
    ASSERT_EQ(LITE_destroy_network(destroy_network), 0);
    LITE_CAPI_CHECK(LITE_destroy_network(c_network));
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
