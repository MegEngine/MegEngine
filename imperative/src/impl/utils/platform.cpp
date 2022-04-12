#include "megbrain/imperative/utils/platform.h"

#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#include <memory>
#endif

using namespace mgb;
using namespace imperative;

/*
 * demangle typeid, see
 * http://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
 */
std::string mgb::imperative::demangle(std::string mangled) {
#ifdef __GNUG__
    int status = -1;
    std::unique_ptr<char, void (*)(void*)> res{
            abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status), std::free};
    return (status == 0) ? res.get() : mangled;
#else
    return mangled;
#endif
}
