#include "test/cuda/fixture.h"

#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, TRANSPOSE) {
    Checker<Transpose> checker(handle_cuda());
    checker.execs({{17, 40}, {40, 17}});
    checker.exec(TensorLayoutArray{
            TensorLayout({17, 40}, {50, 1}, dtype::Float32()),
            TensorLayout({40, 17}, {50, 1}, dtype::Float32())});
    checker.exec(TensorLayoutArray{
            TensorLayout({17, 40}, {50, 1}, dtype::Float16()),
            TensorLayout({40, 17}, {50, 1}, dtype::Float16())});
    checker.exec(TensorLayoutArray{
            TensorLayout({40, 17}, {50, 1}, dtype::Float16()),
            TensorLayout({17, 40}, {50, 1}, dtype::Float16())});
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
