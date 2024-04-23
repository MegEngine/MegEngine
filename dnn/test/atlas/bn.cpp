#include "test/cuda/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"

#include "megdnn/oprs.h"
#include "megdnn/oprs/nn.h"
#include "test/atlas/fixture.h"

#include "test/common/bn.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm.h"
#include "aclnnop/aclnn_batch_norm_backward.h"

namespace megdnn {
namespace test {

using namespace batch_normalization;

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    //   return ret);
    ret = aclrtSetDevice(deviceId);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n",
    //   ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR:
    //   %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR:
    //   %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR:
    //   %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
        const std::vector<T>& hostData, const std::vector<int64_t>& shape,
        void** deviceAddr, aclDataType dataType, aclTensor** tensor,
        aclFormat acl_format = aclFormat::ACL_FORMAT_ND) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n",
    //   ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(
            *deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n",
    //   ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0, acl_format,
            shape.data(), shape.size(), *deviceAddr);
    return 0;
}
void print(const char* name, void* outDeviceAddr, const std::vector<int64_t> outShape) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
            resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host
    //   failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%s [%ld] is: %f\n", name, i, resultData[i]);
    }
}
TEST_F(ATLAS, BN_FORWARD_MAIN) {
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    // check根据自己的需要处理
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
    //   return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {
            1,
            2,
            4,
            1,
    };
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> biasShape = {2};
    std::vector<int64_t> rMeanShape = {2};
    std::vector<int64_t> rVarShape = {2};
    std::vector<int64_t> outShape = {
            1,
            2,
            4,
            1,
    };
    std::vector<int64_t> meanShape = {2};
    std::vector<int64_t> varShape = {2};
    void* selfDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* rMeanDeviceAddr = nullptr;
    void* rVarDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* varDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* rMean = nullptr;
    aclTensor* rVar = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* var = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> rMeanHostData = {0, 0};
    std::vector<float> rVarHostData = {1, 1};
    std::vector<float> outHostData(8, 0);
    std::vector<float> meanHostData = {1, 1};
    std::vector<float> varHostData = {1, 1};
    bool training = true;
    double momentum = 0.1;
    double eps = 1e-5;

    // 创建self aclTensor
    ret = CreateAclTensor(
            selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self,
            aclFormat::ACL_FORMAT_NCHW);
    //   CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(
            weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT,
            &weight);
    //   CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(
            biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    //   CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rMean aclTensor
    ret = CreateAclTensor(
            rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT,
            &rMean);
    //   CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rVar aclTensor
    ret = CreateAclTensor(
            rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
    //   CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(
            outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out,
            aclFormat::ACL_FORMAT_NCHW);
    //   CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mean aclTensor
    ret = CreateAclTensor(
            meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    //   CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建var aclTensor
    ret = CreateAclTensor(
            varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
    //   CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNorm接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnBatchNorm第一段接口
    ret = aclnnBatchNormGetWorkspaceSize(
            self, weight, bias, rMean, rVar, training, momentum, eps, out, mean, var,
            &workspaceSize, &executor);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormGetWorkspaceSize failed.
    //   ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR:
        // %d\n", ret); return ret);
    }
    // 调用aclnnBatchNorm第二段接口
    ret = aclnnBatchNorm(workspaceAddr, workspaceSize, executor, stream);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNorm failed. ERROR: %d\n",
    //   ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR:
    //   %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    print("out", outDeviceAddr, outShape);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(rMean);
    aclDestroyTensor(rVar);
    aclDestroyTensor(out);
    aclDestroyTensor(mean);
    aclDestroyTensor(var);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(rMeanDeviceAddr);
    aclrtFree(rVarDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(varDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

TEST_F(ATLAS, BACKWARD_MAIN) {
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    // check根据自己的需要处理
    //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
    //   return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutShape = {1, 2, 4};
    std::vector<int64_t> selfShape = {1, 2, 4};
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> rMeanShape = {2};
    std::vector<int64_t> rVarShape = {2};
    std::vector<int64_t> sMeanShape = {2};
    std::vector<int64_t> sVarShape = {2};
    std::vector<int64_t> gradInShape = {1, 2, 4};
    std::vector<int64_t> gradWeightShape = {2};
    std::vector<int64_t> gradBiasShape = {2};
    void* gradOutDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* rMeanDeviceAddr = nullptr;
    void* rVarDeviceAddr = nullptr;
    void* sMeanDeviceAddr = nullptr;
    void* sVarDeviceAddr = nullptr;
    void* outMaskDeviceAddr = nullptr;
    void* gradInDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;
    void* gradBiasDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* self = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* rMean = nullptr;
    aclTensor* rVar = nullptr;
    aclTensor* sMean = nullptr;
    aclTensor* sVar = nullptr;
    aclBoolArray* outMask = nullptr;
    aclTensor* gradIn = nullptr;
    aclTensor* gradWeight = nullptr;
    aclTensor* gradBias = nullptr;
    std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> rMeanHostData = {0, 0};
    std::vector<float> rVarHostData = {1, 1};
    std::vector<float> sMeanHostData = {0, 0};
    std::vector<float> sVarHostData = {1, 1};
    std::vector<float> gradInHostData(8, 0);
    std::vector<float> gradWeightHostData(2, 0);
    std::vector<float> gradBiasHostData(2, 0);
    ;
    bool training = true;
    double eps = 1e-5;
    // 创建gradOut aclTensor
    ret = CreateAclTensor(
            gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT,
            &gradOut);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建self aclTensor
    ret = CreateAclTensor(
            selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(
            weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT,
            &weight);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rMean aclTensor
    ret = CreateAclTensor(
            rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT,
            &rMean);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rVar aclTensor
    ret = CreateAclTensor(
            rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sMean aclTensor
    ret = CreateAclTensor(
            sMeanHostData, sMeanShape, &sMeanDeviceAddr, aclDataType::ACL_FLOAT,
            &sMean);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sVar aclTensor
    ret = CreateAclTensor(
            sVarHostData, sVarShape, &sVarDeviceAddr, aclDataType::ACL_FLOAT, &sVar);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outMask aclBoolArray
    bool maskData[2] = {true, true};
    outMask = aclCreateBoolArray(&(maskData[0]), 2);
    // 创建gradIn aclTensor
    ret = CreateAclTensor(
            gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT,
            &gradIn);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradWeight aclTensor
    ret = CreateAclTensor(
            gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr,
            aclDataType::ACL_FLOAT, &gradWeight);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradBias aclTensor
    ret = CreateAclTensor(
            gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr,
            aclDataType::ACL_FLOAT, &gradBias);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNormBackward接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnBatchNormBackward第一段接口
    ret = aclnnBatchNormBackwardGetWorkspaceSize(
            gradOut, self, weight, nullptr, nullptr, sMean, sVar, training, eps,
            outMask, gradIn, gradWeight, gradBias, &workspaceSize, &executor);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackwardGetWorkspaceSize
    // failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR:
        // %d\n", ret); return ret);
    }
    // 调用aclnnBatchNormBackward第二段接口
    ret = aclnnBatchNormBackward(workspaceAddr, workspaceSize, executor, stream);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackward failed. ERROR:
    // %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR:
    // %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
            resultData.data(), resultData.size() * sizeof(resultData[0]),
            gradInDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.
    // ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOut);
    aclDestroyTensor(self);
    aclDestroyTensor(weight);
    aclDestroyTensor(rMean);
    aclDestroyTensor(rVar);
    aclDestroyTensor(sMean);
    aclDestroyTensor(sVar);
    aclDestroyBoolArray(outMask);
    aclDestroyTensor(gradIn);
    aclDestroyTensor(gradWeight);
    aclDestroyTensor(gradBias);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(selfDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(rMeanDeviceAddr);
    aclrtFree(rVarDeviceAddr);
    aclrtFree(sMeanDeviceAddr);
    aclrtFree(sVarDeviceAddr);
    aclrtFree(outMaskDeviceAddr);
    aclrtFree(gradInDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
}
std::vector<TestArg> get_atlas_args() {
    std::vector<TestArg> args;
    // Case 1
    // ParamDim: 1 x 1 x H x W
    // N = 3, C = 3
    // for (size_t i = 4; i < 257; i *= 4) {
    //     param::BN param;
    //     param.fwd_mode = param::BN::FwdMode::TRAINING;
    //     param.param_dim = param::BN::ParamDim::DIM_11HW;
    //     param.avg_factor = 1.f;
    //     args.emplace_back(
    //             param, TensorShape{2, 3, i, i}, TensorShape{1, 1, i, i},
    //             dtype::Float32());
    //     args.emplace_back(
    //             param, TensorShape{2, 3, i, i}, TensorShape{1, 1, i, i},
    //             dtype::Float16());
    // }

    // case 2: 1 x C x 1 x 1

    for (size_t i = 4; i < 10; i *= 4) {
        param::BN param;
        param.fwd_mode = param::BN::FwdMode::TRAINING;
        param.param_dim = param::BN::ParamDim::DIM_1C11;
        args.emplace_back(
                param, TensorShape{3, 3, i, i}, TensorShape{1, 3, 1, 1},
                dtype::Float32());
        args.emplace_back(
                param, TensorShape{3, 3, i, i}, TensorShape{1, 3, 1, 1},
                dtype::Float16());
    }
    param::BN param;
    param.fwd_mode = param::BN::FwdMode::TRAINING;
    param.param_dim = param::BN::ParamDim::DIM_1C11;
    int C = 6;
    args.emplace_back(
            param, TensorShape{3, C, 3, 3}, TensorShape{1, C, 1, 1}, dtype::Float32());
    args.emplace_back(
            param, TensorShape{3, C, 3, 3}, TensorShape{1, C, 1, 1}, dtype::Float16());

    // case 3: 1 x 1 x 1 x C

    // for (size_t i = 4; i < 257; i *= 4) {
    //     param::BN param;
    //     param.fwd_mode = param::BN::FwdMode::TRAINING;
    //     param.param_dim = param::BN::ParamDim::DIM_111C;
    //     args.emplace_back(
    //             param, TensorShape{3, i, i, 3}, TensorShape{1, 1, 1, 3},
    //             dtype::Float32());
    //     args.emplace_back(
    //             param, TensorShape{3, i, i, 3}, TensorShape{1, 1, 1, 3},
    //             dtype::Float16());
    // }

    return args;
}

TEST_F(ATLAS, BN_FORWARD) {
    std::vector<TestArg> args = get_atlas_args();
    Checker<BNForward> checker(handle_atlas());
    for (auto&& arg : args) {
        for (int i = 0; i < 8; ++i) {
            checker.set_dtype(i, dtype::Float32());
        }
        checker.set_dtype(0, arg.dtype);
        checker.set_dtype(8, arg.dtype);
        checker.set_epsilon(1e-3).set_param(arg.param);
        for (bool need_statistic : {true})
            checker.exec({
                    arg.src,
                    arg.param_shape,                                      // bn_scale
                    arg.param_shape,                                      // bn_bias
                    need_statistic ? arg.param_shape : TensorShape({0}),  // mean
                    need_statistic ? arg.param_shape : TensorShape({0}),  // variance
                    arg.param_shape,                                      // batch_mean
                    arg.param_shape,  // batch_inv_variance
                    {1},              // reserve
                    arg.src           // dst
            });
    }
}

TEST_F(ATLAS, BN_BACKWARD) {
    std::vector<TestArg> args = get_atlas_args();
    Checker<BNBackward> checker(handle_atlas());
    UniformFloatRNG ui_rng{.1f, 1000.f};
    checker.set_rng(3, &ui_rng);

    for (auto&& arg : args) {
        for (int i = 0; i < 9; ++i) {
            checker.set_dtype(i, dtype::Float32());
        }
        checker.set_dtype(0, arg.dtype)    // x
                .set_dtype(1, arg.dtype)   // dy
                .set_dtype(8, arg.dtype);  // dx
        checker.set_epsilon(1e-3).set_param(arg.param).exec(
                {arg.src,
                 arg.src,
                 arg.param_shape,
                 arg.param_shape,
                 arg.param_shape,
                 {1},
                 arg.param_shape,
                 arg.param_shape,
                 arg.src});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
