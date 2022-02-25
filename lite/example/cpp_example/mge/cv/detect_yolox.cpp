#include <thread>
#include "example.h"
#if LITE_BUILD_WITH_MGE
#include <cstdio>

#include "misc.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NMS_THRESH       0.25
#define BBOX_CONF_THRESH 0.6

constexpr int INPUT_W = 640;
constexpr int INPUT_H = 640;

using namespace lite;
using namespace example;

namespace {

void preprocess_image(
        uint8_t* image, const int width, const int height, const int channel,
        std::shared_ptr<Tensor> tensor) {
    auto layout = tensor->get_layout();
    for (size_t i = 0; i < layout.ndim; i++) {
        printf("model input shape[%zu]=%zu \n", i, layout.shapes[i]);
    }

    //! resize to target shape
    float r = std::min(INPUT_W / (width * 1.0), INPUT_H / (height * 1.0));
    int unpad_w = r * width;
    int unpad_h = r * height;

    std::shared_ptr<std::vector<uint8_t>> resize_int8 =
            std::make_shared<std::vector<uint8_t>>(unpad_w * unpad_h * channel);
    stbir_resize_uint8(
            image, width, height, 0, resize_int8->data(), unpad_w, unpad_h, 0, channel);

    std::shared_ptr<std::vector<uint8_t>> padded;
    if (unpad_h != INPUT_H || unpad_w != INPUT_W) {
        padded = std::make_shared<std::vector<uint8_t>>(
                INPUT_H * INPUT_W * channel, 114);
        for (int h = 0; h < unpad_h; h++) {
            for (int w = 0; w < unpad_w; w++) {
                for (int c = 0; c < channel; c++) {
                    (*padded)[h * INPUT_W * channel + w * channel + c] =
                            (*resize_int8)[h * unpad_w * channel + w * channel + c];
                }
            }
        }
    } else {
        padded = resize_int8;
    }

    tensor->set_layout({{1, 3, 640, 640}, 4});

    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};

    //! convert form rgb to bgr, relayout from hwc to chw, normalization copy to tensor
    float* in_data = static_cast<float*>(tensor->get_memory_ptr());
    size_t pixels = INPUT_H * INPUT_W;
    for (size_t i = 0; i < pixels; i++) {
        in_data[i] = (padded->at(i * channel + 0) / 255.0f - mean[0]) / std[0];
        in_data[i + 1 * pixels] =
                (padded->at(i * channel + 1) / 255.0f - mean[1]) / std[1];
        in_data[i + 2 * pixels] =
                (padded->at(i * channel + 2) / 255.0f - mean[2]) / std[2];
    }
}

struct Rect {
    float x;
    float y;
    float height;
    float width;

    float area() const { return height * width; }

    Rect operator&(Rect other) const {
        Rect ret;
        float x_start = std::max(x, other.x);
        float x_end = std::min(x + width, other.width);
        ret.x = x_start;
        ret.width = (x_end - x_start) > 0 ? x_end - x_start : 0;

        float y_start = std::max(y, other.y);
        float y_end = std::min(y + height, other.height);
        ret.y = y_start;
        ret.height = (y_end - y_start) > 0 ? y_end - y_start : 0;
        return ret;
    }
};

struct Object {
    Rect rect;
    int label;
    float prob;
};

struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(
        const int target_size, std::vector<int>& strides,
        std::vector<GridAndStride>& grid_strides) {
    for (auto stride : strides) {
        int num_grid = target_size / stride;
        for (int g1 = 0; g1 < num_grid; g1++) {
            for (int g0 = 0; g0 < num_grid; g0++) {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

static void generate_yolox_proposals(
        std::vector<GridAndStride> grid_strides, const float* feat_ptr,
        float prob_threshold, std::vector<Object>& objects) {
    const int num_class = 80;
    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * 85;

        float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
        float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
        float w = exp(feat_ptr[basic_pos + 2]) * stride;
        float h = exp(feat_ptr[basic_pos + 3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_ptr[basic_pos + 4];
        for (int class_idx = 0; class_idx < num_class; class_idx++) {
            float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold) {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        }  // class loop

    }  // point anchor loop
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }
    if (left < j)
        qsort_descent_inplace(faceobjects, left, j);
    if (i < right)
        qsort_descent_inplace(faceobjects, i, right);
}

void qsort_descent_inplace(std::vector<Object>& objects) {
    if (objects.empty())
        return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

inline float intersection_area(const Object& a, const Object& b) {
    Rect inter = a.rect & b.rect;
    return inter.area();
}

void nms_sorted_bboxes(
        const std::vector<Object>& faceobjects, std::vector<int>& picked,
        float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void decode_outputs(
        const float* prob, std::vector<Object>& objects, float scale, const int img_w,
        const int img_h) {
    std::vector<Object> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(INPUT_W, strides, grid_strides);
    generate_yolox_proposals(grid_strides, prob, BBOX_CONF_THRESH, proposals);
    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

void draw_objects(
        uint8_t* image, int width, int height, int channel,
        const std::vector<Object>& objects) {
    (void)image;
    (void)width;
    (void)height;
    (void)channel;
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        printf("Object: %d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
               obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
    }
}

bool detect_yolox(const Args& args) {
    std::string network_path = args.model_path;
    std::string input_path = args.input_path;

    int width, height, channel;
    uint8_t* image = stbi_load(input_path.c_str(), &width, &height, &channel, 0);
    printf("Input image %s with height=%d, width=%d, channel=%d\n", input_path.c_str(),
           width, height, channel);

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(network_path);
    //! set input data to input tensor

    auto input_tensor = network->get_io_tensor("data");

    preprocess_image(image, width, height, channel, input_tensor);

    network->forward();
    network->wait();

    float* predict_ptr =
            static_cast<float*>(network->get_output_tensor(0)->get_memory_ptr());

    float scale = std::min(INPUT_W / (width * 1.0), INPUT_H / (height * 1.0));
    std::vector<Object> objects;
    decode_outputs(predict_ptr, objects, scale, width, height);

    draw_objects(image, width, height, channel, objects);

    stbi_image_free(image);
    return 0;
}
}  // namespace

REGIST_EXAMPLE("detect_yolox", detect_yolox);

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
