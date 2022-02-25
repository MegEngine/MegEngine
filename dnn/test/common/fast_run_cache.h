#pragma once

#include "megdnn/oprs.h"
#include "src/common/hash_ct.h"

#include <unordered_map>

namespace megdnn {
namespace test {
class FastRunCache {
    struct SearchItemStorage {
        std::string data_hold;
        size_t hash = 0;

        SearchItemStorage(const Algorithm::SearchItem& item);

        SearchItemStorage& init_hash() {
            hash = XXHash64CT::hash(data_hold.data(), data_hold.size(), 20201225);
            return *this;
        }

        bool operator==(const SearchItemStorage& rhs) const {
            return data_hold == rhs.data_hold;
        }

        struct Hash {
            size_t operator()(const SearchItemStorage& s) const { return s.hash; }
        };
    };

    std::unordered_map<
            SearchItemStorage, Algorithm::Info::Desc, SearchItemStorage::Hash>
            m_cache;

public:
    Algorithm::Info::Desc get(const Algorithm::SearchItem& key);
    void put(const Algorithm::SearchItem& key, const Algorithm::Info::Desc& val);
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
