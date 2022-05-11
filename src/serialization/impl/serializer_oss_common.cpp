#if MGB_ENABLE_FBS_SERIALIZATION

#include "serializer_oss_common.h"

namespace mgb {
namespace serialization {

bool is_fbs_file(InputFile& file) {
    //! check whether the model format is flatbuffer v2
    uint64_t magic_with_reserved = 0;
    file.read(&magic_with_reserved, sizeof(magic_with_reserved));
    file.skip(-sizeof(magic_with_reserved));
    return (magic_with_reserved == MGB_MAGIC) || (magic_with_reserved == MAGIC_V0);
}

void check_tensor_value_valid(const std::string& name, const HostTensorND& tensor) {
    bool cond_normal = tensor.layout().format.is_default() &&
                       tensor.layout().is_physical_contiguous();
    bool cond_lowbit = tensor.layout().dtype.is_quantized_lowbit() &&
                       tensor.layout().format.is_lowbit_aligned() &&
                       tensor.layout().is_contiguous();
    mgb_assert(
            cond_normal || cond_lowbit, "non-contiguous tensor: name=%s layout=%s",
            name.c_str(), tensor.layout().to_string().c_str());
    if (tensor.dtype() == dtype::Float32()) {
        auto ptr = tensor.ptr<float>();
        for (size_t i = 0, it = tensor.shape().total_nr_elems(); i < it; ++i) {
            if (!std::isfinite(ptr[i])) {
                mgb_log_warn("invalid tensor value in %s: %g", name.c_str(), ptr[i]);
                break;
            }
        }
    }
}

}  // namespace serialization
}  // namespace mgb

#endif
