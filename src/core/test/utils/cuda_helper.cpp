#include "megbrain/utils/cuda_helper.h"
#include "megbrain/test/helper.h"
#include "megbrain_build_config.h"

#if MGB_CUDA
TEST(TestUtils, TestCudaIncludePath) {
    auto paths = mgb::get_cuda_include_path();
    int available = 0;
    for (auto path : paths) {
        FILE* file = fopen((path + "/cuda.h").c_str(), "r");
        if (file) {
            available++;
            fclose(file);
        }
    }
    mgb_assert(available, "no available cuda include path found!");
}
#endif