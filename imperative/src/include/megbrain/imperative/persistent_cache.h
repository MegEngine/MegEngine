#pragma once

#include <memory>
#include "megbrain/utils/persistent_cache.h"

namespace mgb::imperative::persistent_cache {

class ExtendedPersistentCache : public mgb::PersistentCache {
public:
    virtual bool valid() const = 0;
    virtual std::optional<size_t> clear() = 0;
    virtual void flush() = 0;

    static std::shared_ptr<ExtendedPersistentCache> make_from_config(
            std::string type, std::unordered_map<std::string, std::string> args,
            std::string& err_msg);
};

}  // namespace mgb::imperative::persistent_cache
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
