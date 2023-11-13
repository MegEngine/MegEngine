#include <algorithm>

int last_pow2(unsigned int n);

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}