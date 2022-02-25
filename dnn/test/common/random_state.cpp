#include "test/common/random_state.h"

namespace megdnn {
namespace test {

const int RandomState::m_seed;
RandomState RandomState::m_instance = RandomState();

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
