/**
 * \file src/core/test/utils/thin.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/thin/hash_table.h"
#include "megbrain/utils/thin/nullable_hash_map.h"
#include "megbrain/test/helper.h"

using namespace mgb::thin_hash_table;
using mgb::NullableHashMap;

static_assert(ValueTrait<int>::can_embed_in_pair, "bad impl");
static_assert(ValueTrait<void*>::can_embed_in_pair, "bad impl");
static_assert(!ValueTrait<std::pair<int, void*>>::can_embed_in_pair,
              "bad impl");

namespace {
    size_t map_val_inst = 0;

    template<int extra_size>
    class MapVal {
        uint8_t m_data = 0;
        char _[extra_size];

        public:

            MapVal() {
                ++ map_val_inst;
            }

            MapVal(const MapVal &r):
                MapVal()
            {
                *this = r;
            }

            MapVal(MapVal &&r):
                MapVal()
            {
                *this = r;
            }

            MapVal(uint8_t val):
                MapVal()
            {
                m_data = val;
            }

            MapVal& operator = (const MapVal &) = default;
            MapVal& operator = (MapVal &&) = default;

            ~MapVal() {
                -- map_val_inst;
            }

            uint8_t get() {
                return m_data;
            }
    };

    class MapValInplaceOnly: public mgb::NonCopyableObj {
        int m_val;

        public:
            MapValInplaceOnly(int n):
                m_val{n}
            {
            }

            int get() {
                return m_val;
            }
    };

    template<class Val>
    void test_hash_map() {
        ThinHashMap<int, Val> map;
        ASSERT_TRUE(map.empty());
        map[2] = 3;
        ASSERT_FALSE(map.empty());
        ASSERT_TRUE(map.emplace(1, 2).second);
        ASSERT_FALSE(map.insert({2, Val(2)}).second);
        ASSERT_EQ(2u, map.insert({1, Val(0)}).first->second.get());
        ASSERT_EQ(1, map.insert({1, Val(0)}).first->first);
        ASSERT_EQ(2u, map[1].get());
        ASSERT_EQ(3u, map[2].get());

        ASSERT_EQ(2u, map_val_inst);
        ASSERT_EQ(2u, map.size());

        for (auto &&i: map)
            i.second = 5;
        ASSERT_EQ(5u, map[2].get());

        ASSERT_EQ(0u, map[-1].get());
        ASSERT_EQ(map_val_inst, map.size());

        ASSERT_EQ(5u, map.at(2).get());
        map.at(2) = 3;
        ASSERT_EQ(3u, map.at(2).get());
        ASSERT_EQ(1u, map.count(1));
        ASSERT_EQ(0u, map.count(12));
        ASSERT_EQ(map.end(), map.find(12));
        ASSERT_NE(map.end(), map.find(1));
        {
            auto next = std::next(map.find(2));
            ASSERT_EQ(next, map.erase(map.find(2)));
        }
        ASSERT_EQ(0u, map.erase(12));
        ASSERT_EQ(1u, map.erase(-1));
        ASSERT_EQ(1u, map_val_inst);
        ASSERT_EQ(1u, map.size());
        ASSERT_EQ(5u, map.at(1).get());

        map.clear();
        ASSERT_EQ(0u, map.size());
        ASSERT_TRUE(map.empty());
        ASSERT_EQ(map_val_inst, map.size());

        map[0] = 2;
        map.find(0)->second = 3;
        ASSERT_EQ(3u, map[0].get());
        // clear by dtor
    }

    struct IncompleteValue {
        class Value;
        ThinHashMap<int, Value> map;

        void run();
    };

    class IncompleteValue::Value {
        static int sm_inst;
        int m_v;

    public:
        Value(int v = 0) : m_v{v} { ++sm_inst; }
        ~Value() { --sm_inst; }
        int v() const { return m_v; }
        static int inst() { return sm_inst; }
    };
    int IncompleteValue::Value::sm_inst;

    void IncompleteValue::run() {
        map[0] = 23;
        map[1] = 45;
        ASSERT_EQ(2u, map.size());
        ASSERT_EQ(23, map[0].v());
        ASSERT_EQ(0, map[3].v());
        ASSERT_EQ(3u, map.size());
        ASSERT_EQ(1u, map.erase(0));
        ASSERT_EQ(2u, map.size());
        ASSERT_EQ(2, Value::inst());
    }
}

TEST(TestThinHashTable, ThinHashSet) {
    ThinHashSet<int> set;
    ASSERT_EQ(0u, set.size());
    ASSERT_EQ(true, set.empty());
    set.insert(2);
    set.insert(3);
    ASSERT_EQ(1u, set.count(2));
    ASSERT_EQ(1u, set.count(3));
    ASSERT_EQ(0u, set.count(1));
    ASSERT_EQ(2u, set.size());
    ASSERT_FALSE(set.empty());

    std::vector<int> get;
    for (int i: set)
        get.push_back(i);
    ASSERT_EQ(2u, get.size());

    set.emplace(4);
    // set: {2, 3, 4}
    ASSERT_EQ(0u, set.erase(0));
    ASSERT_EQ(1u, set.erase(3));
    ASSERT_EQ(2u, set.size());
    set.erase(set.find(4));
    ASSERT_EQ(1u, set.size());
    ASSERT_EQ(2, *set.begin());

    set.clear();
    ASSERT_TRUE(set.empty());
}

TEST(TestThinHashTable, InitializerList) {
    ThinHashMap<int, int> m{{1, 2}, {3, 4}};
    ASSERT_EQ(m.at(1), 2);
    ASSERT_EQ(m.at(3), 4);
    ASSERT_THROW(m.at(2), std::exception);
}

TEST(TestThinHashTable, ThinHashMapSmallVal) {
    ASSERT_EQ(0u, map_val_inst);
    test_hash_map<MapVal<1>>();
    ASSERT_EQ(0u, map_val_inst);
}

TEST(TestThinHashTable, ThinHashMapBigVal) {
    ASSERT_EQ(0u, map_val_inst);
    test_hash_map<MapVal<sizeof(void*)>>();
    ASSERT_EQ(0u, map_val_inst);
}

TEST(TestThinHashTable, ThinHashMapInplaceOnly) {
    ThinHashMap<int, MapValInplaceOnly> map;
    map.emplace(std::piecewise_construct,
            std::forward_as_tuple(1), std::forward_as_tuple(23));
    ASSERT_EQ(23, map.at(1).get());
}

TEST(TestThin, NullableHashMap) {
    ASSERT_EQ(0u, map_val_inst);
    {
        NullableHashMap<int, MapVal<2>> map;
        ASSERT_EQ(nullptr, map.get(2));
        map.set(2, map.alloc(3));
        ASSERT_EQ(1u, map_val_inst);
        ASSERT_EQ(nullptr, map.get(3));
        ASSERT_EQ(3u, map.get(2)->get());

        map.set(2, map.alloc(5));
        ASSERT_EQ(1u, map_val_inst);
        ASSERT_EQ(5u, map.get(2)->get());

        map.clear();
        ASSERT_EQ(0u, map_val_inst);
        map.set(2, map.alloc(5));
        ASSERT_EQ(1u, map_val_inst);
        map.set(3, map.alloc(5));
        ASSERT_EQ(2u, map_val_inst);
    }
    ASSERT_EQ(0u, map_val_inst);
}

TEST(TestThin, HashMapIncompleteValue) {
    IncompleteValue{}.run();
    ASSERT_EQ(0, IncompleteValue::Value::inst());
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

