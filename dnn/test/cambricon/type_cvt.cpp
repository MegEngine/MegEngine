#include "megdnn/oprs.h"
#include "test/cambricon/fixture.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CAMBRICON, TYPE_CVT) {
    UniformFloatRNG init(0, 20);
    std::vector<DType> dtypes = {dtype::Float32(), dtype::Int32(), dtype::Int16(),
                                 dtype::Int8(),    dtype::Uint8(), dtype::Float16()};
    for (auto sdtype : dtypes)
        for (auto ddtype : dtypes) {
            if ((sdtype == dtype::Int32() || sdtype == dtype::Int8()) &&
                        ddtype == dtype::Uint8() ||
                sdtype == dtype::Int16() &&
                        (ddtype == dtype::Int8() || ddtype == dtype::Uint8()) ||
                sdtype == dtype::Uint8() &&
                        (ddtype == dtype::Int16() || ddtype == dtype::Int8()))
                continue;

            TensorLayout src({10, 10}, sdtype), dst({10, 10}, ddtype);
            Checker<TypeCvt> checker(handle_cambricon());
            checker.set_rng(0, &init).exec(TensorLayoutArray{src, dst});

            TensorLayout non_contig_src(
                    {1, 96, 64, 120}, {96 * 64 * 128, 64 * 128, 128, 1}, sdtype);
            TensorLayout non_contig_dst({1, 96, 64, 120}, ddtype);
            checker.exec(TensorLayoutArray{non_contig_src, non_contig_dst});
        }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen