#if MGB_ENABLE_FBS_SERIALIZATION

#include "megbrain/serialization/serializer.h"
#include "megbrain/version.h"

namespace mgb {
namespace serialization {

constexpr uint32_t MGB_VERSION = (MGE_MAJOR * 1000 + MGE_MINOR) * 100 + MGE_PATCH;

constexpr uint32_t MGB_MAGIC = 0x4342474D;

// In order to maintain compatibility and to allow old models to be loaded, we keep
// the old magic(MAGIC_V0) value and creat a new magic(MGB_MAGIC)
constexpr uint32_t MAGIC_V0 = 0x5342474D;

void check_tensor_value_valid(const std::string& name, const HostTensorND& tensor);

template <typename T>
bool contains_any_in_set(const SmallVector<T>& list, const ThinHashSet<T>& set) {
    for (const auto& x : list) {
        if (set.count(x)) {
            return true;
        }
    }
    return false;
}

}  // namespace serialization
}  // namespace mgb

#endif
