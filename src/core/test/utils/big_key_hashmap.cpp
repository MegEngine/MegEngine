/**
 * \file src/core/test/utils/big_key_hashmap.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/big_key_hashmap.h"
#include "megbrain/test/helper.h"

using namespace mgb;
using namespace big_key_hash_map;

namespace {
int nr_key_copy = 0, refcnt = 0;

class TestError final : public std::exception {};

class RefCnt {
public:
    RefCnt() { ++refcnt; }
    ~RefCnt() { --refcnt; }
};

class Key {
    RefCnt m_refcnt;
    const int m_val;

public:
    Key(int val) : m_val{val} {}

    Key(const Key& rhs) : m_val{rhs.m_val} {
        if (m_val == -23) {
            mgb_throw_raw(TestError{});
        }
        ++nr_key_copy;
    }

    bool operator==(const Key& rhs) const { return m_val == rhs.m_val; }

    struct Eq {
        template <typename T>
        static bool eq(const T& x, const T& y) {
            return x == y;
        }
    };
    struct Hash : public Eq {
        static size_t hash(const Key& x) { return x.m_val; }
        static size_t hash(int x) { return x; }
    };
    struct HashConflict : public Eq {
        static size_t hash(const Key&) { return 0; }
        static size_t hash(int) { return 0; }
    };
};

template <class Hash>
void run_multi_key() {
    nr_key_copy = 0;
    {
        BigKeyHashMap<int, Hash, Ref<Key>, Copy<int>, Ref<Key>> map;
        auto v1 = map.get(1, 1, 1);
        ASSERT_TRUE(v1.first);
        ASSERT_EQ(2, nr_key_copy);
        ASSERT_FALSE(map.get(1, 1, 1).first);
        ASSERT_EQ(v1.second, map.get(1, 1, 1).second);
        ASSERT_EQ(2, nr_key_copy);

        auto v2 = map.get(1, 1, 2);
        ASSERT_TRUE(v2.first);
        ASSERT_EQ(4, nr_key_copy);
        ASSERT_FALSE(map.get(1, 1, 2).first);
        ASSERT_EQ(v2.second, map.get(1, 1, 2).second);
        ASSERT_NE(v1.second, v2.second);
        ASSERT_EQ(4, nr_key_copy);
        ASSERT_EQ(4, refcnt);

        ThinHashSet<int*> vals;
        for (int run = 0; run < 3; ++run) {
            for (int i = 3; i < 5; ++i) {
                for (int j = 3; j < 5; ++j) {
                    for (int k = 3; k < 5; ++k) {
                        auto ins = map.get(i, j, k);
                        if (run) {
                            ASSERT_FALSE(ins.first);
                            ASSERT_EQ(20, nr_key_copy);
                            ASSERT_EQ(20, refcnt);
                        } else {
                            ASSERT_TRUE(ins.first);
                        }
                        vals.insert(ins.second);
                    }
                }
            }
        }
        ASSERT_EQ(8u, vals.size());
    }
    ASSERT_EQ(0, refcnt);
}

}  // anonymous namespace

TEST(TestBigKeyHashMap, Simple) {
    nr_key_copy = 0;
    {
        BigKeyHashMap<int, Key::Hash, Ref<Key>> map;
        auto v1 = map.get(1);
        ASSERT_TRUE(v1.first);
        ASSERT_EQ(1, nr_key_copy);
        ASSERT_FALSE(map.get(1).first);
        ASSERT_EQ(v1.second, map.get(1).second);
        ASSERT_EQ(1, nr_key_copy);

        auto v2 = map.get(2);
        ASSERT_TRUE(v2.first);
        ASSERT_EQ(2, nr_key_copy);
        ASSERT_FALSE(map.get(2).first);
        ASSERT_EQ(v2.second, map.get(2).second);
        ASSERT_NE(v1.second, v2.second);
        ASSERT_EQ(2, nr_key_copy);
        ASSERT_EQ(2, refcnt);
    }
    ASSERT_EQ(0, refcnt);
}

TEST(TestBigKeyHashMap, MultiKey) {
    run_multi_key<Key::Hash>();
}

TEST(TestBigKeyHashMap, MultiKeyHashConflict) {
    run_multi_key<Key::HashConflict>();
}

#if MGB_ENABLE_EXCEPTION
TEST(TestBigKeyHashMap, ExcSafe) {
    nr_key_copy = 0;
    {
        BigKeyHashMap<int, Key::Hash, Ref<Key>, Ref<Key>> map;
        map.get(2, 3);
        ASSERT_EQ(2, refcnt);
        ASSERT_EQ(2, nr_key_copy);
        ASSERT_THROW(map.get(-23, 1), TestError);
        ASSERT_THROW(map.get(1, -23), TestError);
        ASSERT_EQ(1u, map.size());
        ASSERT_EQ(2, refcnt);
        ASSERT_EQ(3, nr_key_copy);
    }
    ASSERT_EQ(0, refcnt);
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

