#include "megbrain/common.h"

#include "megbrain/comp_node_env.h"
#include "megbrain/custom/data_adaptor.h"
#include "megbrain/custom/platform/custom_cuda.h"

using namespace mgb;

namespace custom {

const CudaRuntimeArgs get_cuda_runtime_args(const RuntimeArgs& rt_args) {
    mgb_assert(
            rt_args.device().enumv() == DeviceEnum::cuda,
            "devive type should be cuda.");
    const CompNodeEnv& env =
            CompNodeEnv::from_comp_node(to_builtin<CompNode, Device>(rt_args.device()));
    const CompNodeEnv::CudaEnv& cuda_env = env.cuda_env();
    return {cuda_env.device, cuda_env.stream};
}

}  // namespace custom
