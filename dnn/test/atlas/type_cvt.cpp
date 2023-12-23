#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(ATLAS, TYPE_CVT) {
    UniformFloatRNG init(0, 20);
    std::vector<DType> dtypes = {dtype::Float32(), dtype::Float16(), dtype::Int32(),
                                 dtype::Int16(),   dtype::Uint16(),  dtype::Int8(),
                                 dtype::Uint8(),   dtype::BFloat16()};
    for (auto sdtype : dtypes)
        for (auto ddtype : dtypes) {
            TensorLayout src({10, 10}, sdtype), dst({10, 10}, ddtype);
            Checker<TypeCvt> checker(handle_atlas());
            checker.set_rng(0, &init).exec(TensorLayoutArray{src, dst});
        }
}

TEST_F(ATLAS, TYPE_CVT_NON_CONTIG_SRC) {
    UniformFloatRNG init(0, 20);
    std::vector<DType> sdtypes = {dtype::Float32(), dtype::Float16(), dtype::Int32(),
                                  dtype::Int16(),   dtype::Uint16(),  dtype::Int8(),
                                  dtype::Uint8(),   dtype::BFloat16()};
    std::vector<DType> ddtypes = {dtype::Float32(), dtype::Float16(), dtype::Int32(),
                                  dtype::Int16(),   dtype::Uint16(),  dtype::Int8(),
                                  dtype::Uint8(),   dtype::BFloat16()};
    for (auto sdtype : sdtypes)
        for (auto ddtype : ddtypes) {
            // TODO: this case may introduce strange error and skip it temporarily.
            if (sdtype.enumv() == DTypeEnum::Uint16 &&
                ddtype.enumv() == DTypeEnum::Float16) {
                continue;
            }
            TensorLayout non_contig_src(
                    {1, 96, 64, 120}, {96 * 64 * 128, 64 * 128, 128, 1}, sdtype);
            TensorLayout contig_dst({1, 96, 64, 120}, ddtype);
            Checker<TypeCvt> checker(handle_atlas());
            checker.set_rng(0, &init).exec(
                    TensorLayoutArray{non_contig_src, contig_dst});
        }
}

// vim: syntax=cpp.doxygen
