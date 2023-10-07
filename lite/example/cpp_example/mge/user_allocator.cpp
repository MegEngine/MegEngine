#include "example.h"
#if LITE_BUILD_WITH_MGE
using namespace lite;
using namespace example;

namespace {
class CheckAllocator : public lite::Allocator {
public:
    //! allocate memory of size in the given device with the given align
    void* allocate(LiteDeviceType, int, size_t size, size_t align) override {
#if defined(WIN32) || defined(_WIN32)
        return _aligned_malloc(size, align);
#elif defined(__ANDROID__) || defined(ANDROID) || defined(__OHOS__)
        return memalign(align, size);
#else
        void* ptr = nullptr;
        auto err = posix_memalign(&ptr, align, size);
        if (!err) {
            printf("failed to malloc %zu bytes with align %zu", size, align);
        }
        return ptr;
#endif
    };

    //! free the memory pointed by ptr in the given device
    void free(LiteDeviceType, int, void* ptr) override {
#if defined(WIN32) || defined(_WIN32)
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
    };
};

bool config_user_allocator(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    auto allocator = std::make_shared<CheckAllocator>();

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();

    Runtime::set_memory_allocator(network, allocator);

    network->load_model(network_path);

    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    //! copy or forward data to network
    size_t length = input_tensor->get_tensor_total_size_in_byte();
    void* dst_ptr = input_tensor->get_memory_ptr();
    auto src_tensor = parse_npy(input_path);
    void* src = src_tensor->get_memory_ptr();
    memcpy(dst_ptr, src, length);

    //! forward
    network->forward();
    network->wait();

    //! get the output data or read tensor set in network_in
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    void* out_data = output_tensor->get_memory_ptr();
    size_t out_length = output_tensor->get_tensor_total_size_in_byte() /
                        output_tensor->get_layout().get_elem_size();
    printf("length=%zu\n", length);
    float max = -1.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < out_length; i++) {
        float data = static_cast<float*>(out_data)[i];
        sum += data;
        if (max < data)
            max = data;
    }
    printf("max=%e, sum=%e\n", max, sum);
    return true;
}
}  // namespace

REGIST_EXAMPLE("config_user_allocator", config_user_allocator);

#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
