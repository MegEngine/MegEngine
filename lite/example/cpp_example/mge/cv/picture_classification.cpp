#include <thread>
#include "example.h"
#if LITE_BUILD_WITH_MGE
#include <cstdio>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace lite;
using namespace example;

namespace {

void preprocess_image(std::string pic_path, std::shared_ptr<Tensor> tensor) {
    int width, height, channel;
    uint8_t* image = stbi_load(pic_path.c_str(), &width, &height, &channel, 0);
    printf("Input image %s with height=%d, width=%d, channel=%d\n", pic_path.c_str(),
           width, height, channel);
    auto layout = tensor->get_layout();
    auto pixels = layout.shapes[2] * layout.shapes[3];
    for (size_t i = 0; i < layout.ndim; i++) {
        printf("model input shape[%zu]=%zu \n", i, layout.shapes[i]);
    }
    //! resize to tensor shape
    std::shared_ptr<std::vector<uint8_t>> resize_int8 =
            std::make_shared<std::vector<uint8_t>>(pixels * channel);

    stbir_resize_uint8(
            image, width, height, 0, resize_int8->data(), layout.shapes[2],
            layout.shapes[3], 0, channel);

    stbi_image_free(image);

    //! convert form rgba to bgr, relayout from hwc to chw, normalization copy to tensor
    float* in_data = static_cast<float*>(tensor->get_memory_ptr());
    for (size_t i = 0; i < pixels; i++) {
        in_data[i + 2 * pixels] = (resize_int8->at(i * channel + 0) - 123.675) / 58.395;
        in_data[i + 1 * pixels] = (resize_int8->at(i * channel + 1) - 116.280) / 57.120;
        in_data[i + 0 * pixels] = (resize_int8->at(i * channel + 2) - 103.530) / 57.375;
    }
}

void classfication_process(
        std::shared_ptr<Tensor> tensor, float& score, size_t& class_id) {
    auto layout = tensor->get_layout();
    for (size_t i = 0; i < layout.ndim; i++) {
        printf("model output shape[%zu]=%zu \n", i, layout.shapes[i]);
    }
    size_t nr_data = tensor->get_tensor_total_size_in_byte() / layout.get_elem_size();
    float* data = static_cast<float*>(tensor->get_memory_ptr());
    score = data[0];
    class_id = 0;
    float sum = data[0];
    for (size_t i = 1; i < nr_data; i++) {
        if (score < data[i]) {
            score = data[i];
            class_id = i;
        }
        sum += data[i];
    }
    printf("output tensor sum is %f\n", sum);
}

bool picture_classification(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(network_path);
    //! set input data to input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    //! copy or forward data to network
    preprocess_image(args.input_path, input_tensor);

    printf("Begin forward.\n");
    network->forward();
    network->wait();
    printf("End forward.\n");

    //! get the output data or read tensor set in network_in
    size_t class_id;
    float score;
    auto output_tensor = network->get_output_tensor(0);
    classfication_process(output_tensor, score, class_id);
    printf("Picture %s is class_id %zu, with score %f\n", args.input_path.c_str(),
           class_id, score);
    return 0;
}
}  // namespace

REGIST_EXAMPLE("picture_classification", picture_classification);

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
