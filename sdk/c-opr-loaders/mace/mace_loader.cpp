/**
 * \file sdk/c-opr-loaders/mace/mace_loader.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <numeric>
#include <iostream>

#include "mace/public/mace.h"
#include "extern_c_opr.h"

#define ASSERT(x, msg)                                                       \
    do {                                                                     \
        if (!(x)) {                                                          \
            printf("error at %s:%d %s\n", __FILE__, __LINE__, __FUNCTION__); \
            printf(msg);                                                     \
            __builtin_trap();                                                \
        }                                                                    \
    } while (0)


class MGBOprDescImpl {
    struct UserData {
        std::shared_ptr<mace::MaceEngine> engine;
        size_t nr_inputs, nr_outputs;
        std::vector<std::vector<int64_t>> output_shapes;
        std::vector<std::string> input_names, output_names;
    };

    static UserData* user_data(const MGBOprDesc* self) {
        return static_cast<UserData*>(self->user_data);
    }

    static void release(MGBOprDesc* self) {
        // free all data buffers
        delete user_data(self);
        delete self;
    }

    static size_t hash(const MGBOprDesc* self) {
        return reinterpret_cast<size_t>(self);
    }

    static int is_same(const MGBOprDesc* self, const MGBOprDesc* rhs) {
        return self == rhs;
    }

    static void infer_shape(const MGBOprDesc* self, const MGBTensorShape* input,
                            MGBTensorShape* output) {
        auto ud = user_data(self);

        // infer output shape from user data
        for (size_t i = 0; i < ud->nr_outputs; i++) {
            output[i].ndim = ud->output_shapes[i].size();
            for (size_t j = 0; j < output[i].ndim; j++) {
                output[i].shape[j] = ud->output_shapes[i][j];
            }
        }
    }

    static void infer_dtype(const MGBOprDesc*, const MGBDType* input, MGBDType* output) {
        ASSERT(input[0] == MGB_DTYPE_FLOAT32, "Input dtype is not float32");
        output[0] = MGB_DTYPE_FLOAT32;
    }

    static void execute(const MGBOprDesc* self, const MGBTensor* input,
                        const MGBTensor* output) {
        auto ud = user_data(self);

        // create input and output tensor buffers
        std::map<std::string, mace::MaceTensor> mace_inputs;
        std::map<std::string, mace::MaceTensor> mace_outputs;

        auto mace_data_format = mace::DataFormat::NCHW;
        char *data_format = getenv("DATAFORMAT");
        if (!strcmp(data_format, "NHWC")) {
            mace_data_format = mace::DataFormat::NHWC;
        }

        for (size_t i = 0; i < ud->nr_inputs; ++i) {
            // allocate input
            uint32_t ndim = input[i].layout.shape.ndim;
            auto input_shape = std::vector<int64_t>(input[i].layout.shape.shape,
                                                    input[i].layout.shape.shape + ndim);

            int64_t input_size =
                std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                std::multiplies<uint64_t>());
            auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                                    std::default_delete<float[]>());
            memcpy(buffer_in.get(), input[i].data, input_size * sizeof(float));
            mace_inputs[ud->input_names[i]] =
                mace::MaceTensor(input_shape, buffer_in, mace_data_format);
        }

        for (size_t i = 0; i < ud->nr_outputs; ++i) {
            // allocate output
            uint32_t ndim = output[i].layout.shape.ndim;
            auto output_shape = std::vector<int64_t>(output[i].layout.shape.shape,
                                                     output[i].layout.shape.shape + ndim);

            int64_t output_size =
                std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int64_t>());
            auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                                     std::default_delete<float[]>());
            mace_outputs[ud->output_names[i]] =
                mace::MaceTensor(output_shape, buffer_out, mace_data_format);
        }

        // run the model
        auto status = (ud->engine)->Run(mace_inputs, &mace_outputs);
        ASSERT(status == mace::MaceStatus::MACE_SUCCESS,
               "Error in running mace engine");

        // send computed output to MGB
        int idx = 0;
        for (auto it = mace_outputs.begin(); it != mace_outputs.end(); it++) {
            float* to = &((float *)output[idx++].data)[0];
            to = (it->second).data().get();
        }
    }

public:
    static MGBOprDesc* make(size_t nr_input, const void *buf, size_t buf_len) {
        auto ud = std::make_unique<UserData>();

        std::shared_ptr<mace::MaceEngine> engine;
        mace::DeviceType device_type;

        char *runtime_mode = getenv("RUNTIME");
        if (!strcmp(runtime_mode, "GPU")) {
            device_type = mace::DeviceType::GPU;
        } else {
            device_type = mace::DeviceType::CPU;
        }
        mace::MaceEngineConfig config(device_type);

        // set gpu context, mainly opencl path
        if (device_type == mace::DeviceType::GPU) {
            std::shared_ptr<mace::GPUContext> gpu_context;

            char *opencl_path = getenv("OPENCLPATH");
            ASSERT(opencl_path, "Please set opencl library path");
            std::string storage_path(opencl_path);
            gpu_context = mace::GPUContextBuilder()
                            .SetStoragePath(storage_path)
                            .Finalize();

            config.SetGPUContext(gpu_context);
            config.SetGPUHints(
                static_cast<mace::GPUPerfHint>(mace::GPUPerfHint::PERF_HIGH),
                static_cast<mace::GPUPriorityHint>(mace::GPUPriorityHint::PRIORITY_HIGH));
        }

        std::vector<std::string> input_names, output_names;

        // extract all information from buf

        void *buffer = const_cast<void *>(buf);

        ud->nr_inputs = *reinterpret_cast<uint32_t*>(buffer);
        ud->nr_outputs = *(reinterpret_cast<uint32_t*>(buffer) + 1);

        // interpret input names
        char *name_buf = reinterpret_cast<char*>(buffer) + 8;
        for (size_t i = 0; i < ud->nr_inputs; i++) {
            size_t ilen = *reinterpret_cast<uint32_t*>(name_buf);
            input_names.push_back(std::string(name_buf + 4, ilen));
            name_buf += (ilen + 4);
        }

        // interpret output names
        buffer = name_buf;
        name_buf = reinterpret_cast<char*>(buffer);
        for (size_t i = 0; i < ud->nr_outputs; i++) {
            size_t olen = *reinterpret_cast<uint32_t*>(name_buf);
            output_names.push_back(std::string(name_buf + 4, olen));
            name_buf += (olen + 4);
        }

        ud->input_names = input_names;
        ud->output_names = output_names;

        // interpret output shapes
        buffer = name_buf;
        uint32_t *shape_buf = reinterpret_cast<uint32_t*>(buffer) + 1;
        for (size_t i = 0; i < ud->nr_outputs; i++) {
            size_t olen = *reinterpret_cast<int*>(shape_buf);
            ud->output_shapes.push_back(
                std::vector<int64_t>(shape_buf + 1, shape_buf + olen + 1)
            );
            shape_buf += (olen + 1);
        }

        buffer = shape_buf;
        const size_t model_buf_len = *reinterpret_cast<int*>(buffer);
        unsigned char *model_buf = reinterpret_cast<unsigned char*>(buffer) + 4;

        const size_t param_buf_len = *reinterpret_cast<int*>(model_buf + model_buf_len);
        unsigned char *param_buf = model_buf + model_buf_len + 4;

        // create mace engine
        auto create_engine_status = mace::CreateMaceEngineFromProto(
            model_buf,
            model_buf_len,
            param_buf,
            param_buf_len,
            input_names,
            output_names,
            config,
            &engine
        );
        ASSERT(create_engine_status == mace::MaceStatus::MACE_SUCCESS,
               "Error in creating mace engine");

        ud->engine = engine;

        auto ret = std::make_unique<MGBOprDesc>();
        mgb_init_opr_desc(ret.get(), ud->nr_outputs, "mace");
#define a(n) ret->n = &n;
        MGB_OPR_DESC_FOREACH_MEM_FN(a);
        a(infer_dtype);
#undef a
        ret->user_data = ud.release();
        return ret.release();
    }
};

class MGBOprLoaderImpl {
    static MGBOprDesc* create_desc(size_t nr_input, const void *buf,
            size_t buf_len)
    {
        return MGBOprDescImpl::make(nr_input, buf, buf_len);
    }
public:
    static MGBOprLoader make() {
        return {"mace", create_desc};
    }
};

extern "C" {

// public interface
__attribute__((visibility("default")))
void MGB_C_OPR_INIT_FUNC(const MGBExternCOprApi* (*get_api)(int))
{
    const MGBExternCOprApi* api = get_api(MGB_EXTERN_C_OPR_VERSION);
    ASSERT(api, "Create api failed");
    MGBOprLoader loader = MGBOprLoaderImpl::make();
    api->register_loader(&loader);
}

}  // extern "C"
