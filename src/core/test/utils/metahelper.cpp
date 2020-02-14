/**
 * \file src/core/test/utils/metahelper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/metahelper.h"
#include "megbrain/test/helper.h"

using namespace mgb;

namespace {
    //! for testing Maybe
    class RefCnt : NonCopyableObj {
    public:
        static int cnt;
        RefCnt() { ++cnt; };
        ~RefCnt() { --cnt; }
    };
    int RefCnt::cnt = 0;

    struct TestIncompleteStorage {
        class T;
        IncompleteObjStorage<T, 4, 4> m_t;

        T& t() {
            return m_t.get();
        }
    };

    class TestIncompleteStorage::T {
        int m = 123;

        public:
            static bool dtor;

            int& get() {
                return m;
            }

            ~T() {
                dtor = true;
            }
    };
    bool TestIncompleteStorage::T::dtor = false;

    class UserData: public UserDataContainer::UserData {
        int *m_refcnt;
        MGB_TYPEINFO_OBJ_DECL;

        public:
            UserData(int *refcnt):
                m_refcnt{refcnt}
            {
                ++ *m_refcnt;
            }

            ~UserData() {
                -- *m_refcnt;
            }

    };

    class UserData1: public UserData {
        MGB_TYPEINFO_OBJ_DECL;
        public:
            using UserData::UserData;
    };

    MGB_TYPEINFO_OBJ_IMPL(UserData);
    MGB_TYPEINFO_OBJ_IMPL(UserData1);
}

TEST(TestMetahelper, SmallSort) {
    bool fail = false;

    for (int N = 0; N <= 6; ++ N) {
        std::vector<int> arr(N);
        thin_function<void(int)> gen;
        gen = [&](int p) {
            ASSERT_FALSE(fail);
            if (p == N) {
                fail = true;
                auto s0 = arr, s1 = arr;
                std::sort(s0.begin(), s0.end());
                small_sort(s1.begin(), s1.end());
                for (int i = 0; i < N; ++ i) {
                    ASSERT_EQ(s0[i], s1[i]) << "fail at " << i;
                }
                fail = false;
                return;
            }
            for (int i = 0; i < N; ++ i) {
                arr[p] = i;
                gen(p + 1);
            }
        };
        gen(0);
    }

}

TEST(TestMetahelper, IncompleteStorage) {
    TestIncompleteStorage::T::dtor = false;
    {
        TestIncompleteStorage s;
        auto &&t = s.t();
        ASSERT_EQ(123, t.get());
        t.get() += 1;
        ASSERT_EQ(124, s.t().get());
        ASSERT_FALSE(TestIncompleteStorage::T::dtor);
    }
    ASSERT_TRUE(TestIncompleteStorage::T::dtor);
}

TEST(TestMetahelper, UserDataContainer) {
    int refcnt = 0;

    {
        UserDataContainer ct;
        ASSERT_EQ(nullptr, ct.get_user_data<UserData>().first);
        auto ptr = ct.get_user_data_or_create<UserData>([&](){
                return std::make_shared<UserData>(&refcnt); });
        ASSERT_NE(nullptr, ptr);
        ASSERT_EQ(ptr, ct.get_user_data<UserData>().first[0]);
        ASSERT_EQ(1, refcnt);

        int rm = ct.pop_user_data<UserData>();
        ASSERT_EQ(0, refcnt);
        ASSERT_EQ(rm, 1);
        ASSERT_EQ(nullptr, ct.get_user_data<UserData>().first);

        auto ptr1 = ct.add_user_data<UserData1>(
                std::make_shared<UserData1>(&refcnt));
        ASSERT_EQ(nullptr, ct.get_user_data<UserData>().first);
        ASSERT_EQ(ptr1, ct.get_user_data<UserData1>().first[0]);

        ASSERT_EQ(0, ct.pop_user_data<UserData>());
        ASSERT_EQ(1, refcnt);
    }
    ASSERT_EQ(0, refcnt);
}

/* ======================= begin Maybe ======================= */
#define CHK(v) ASSERT_EQ(v, RefCnt::cnt)

TEST(TestMetahelper, MaybeAssign) {
    // use shared_ptr<> to easily check whether copy/move ctor/opr= functions
    // are correctly called
    // this case tests operator=, invalidate() and emplace()
    auto chk_assign = [](bool move, bool lhs_valid, bool rhs_valid) {
        Maybe<std::shared_ptr<RefCnt>> m0, m1;
        if (lhs_valid) {
            m0.emplace(new RefCnt);
        }
        if (rhs_valid) {
            m1.emplace(new RefCnt);
        }
        ASSERT_EQ(lhs_valid, m0.valid());
        ASSERT_EQ(rhs_valid, m1.valid());

        CHK(lhs_valid + rhs_valid);

        if (move) {
            m0 = std::move(m1);
            if (rhs_valid) {
                ASSERT_EQ(nullptr, m1->get());
            }
        } else {
            m0 = m1;
            if (rhs_valid) {
                ASSERT_NE(nullptr, m1->get());
            }
        }
        ASSERT_EQ(rhs_valid, m0.valid());
        ASSERT_EQ(rhs_valid, m1.valid());
        CHK(rhs_valid);
    };

    for (int i = 0; i < 8; ++i) {
        chk_assign((i >> 2) & 1, (i >> 1) & 1, i & 1);
        CHK(0);
    }
}

TEST(TestMetahelper, MaybeCtor) {
    // test ctor
    {
        Maybe<std::shared_ptr<RefCnt>> m0{new RefCnt}, m1{new RefCnt}, m2{m0},
                m3{std::move(m1)};
        CHK(2);
        ASSERT_NE(nullptr, m0->get());
        ASSERT_EQ(nullptr, m1->get());
        m0 = None;
        ASSERT_FALSE(m0.valid());
        CHK(2);
        m2 = None;
        CHK(1);
    }
    CHK(0);

    // test emplace with zero args; also ensure no object when invalid
    {
        Maybe<RefCnt> x;
        CHK(0);
        x.emplace();
        CHK(1);
        x = None;
        CHK(0);
    }
    CHK(0);

    // test emplace with two args
    {
        Maybe<std::pair<int, int>> y;
        y.emplace(1, 2);
        ASSERT_EQ(std::make_pair(1, 2), y.val());

        // test opr ->
        ASSERT_EQ(2, y->second);
    }
}

TEST(TestMetahelper, MaybeExcept) {
    class T0 {
    public:
        T0(T0&&) {}
    };
    class T1 {
    public:
        T1& operator=(T1&&) { return *this; }
    };
    class T2 {};
    ASSERT_FALSE(std::is_nothrow_move_constructible<T0>::value);
    ASSERT_FALSE(std::is_nothrow_move_assignable<T1>::value);
    ASSERT_TRUE(std::is_nothrow_move_assignable<T2>::value);

    ASSERT_FALSE(std::is_nothrow_move_constructible<Maybe<T0>>::value);
    ASSERT_FALSE(std::is_nothrow_move_assignable<Maybe<T0>>::value);
    ASSERT_FALSE(std::is_nothrow_move_constructible<Maybe<T1>>::value);
    ASSERT_FALSE(std::is_nothrow_move_assignable<Maybe<T1>>::value);
    ASSERT_TRUE(std::is_nothrow_move_constructible<Maybe<T2>>::value);
    ASSERT_TRUE(std::is_nothrow_move_assignable<Maybe<T2>>::value);
}

#undef CHK

/* ======================= end Maybe ======================= */

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

