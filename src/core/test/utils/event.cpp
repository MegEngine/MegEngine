/**
 * \file src/core/test/utils/event.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/utils/event.h"

using namespace mgb;

namespace {
    struct Event0 {
        int x;

        MGB_TYPEINFO_OBJ_DECL;
    };

    struct Event1 {
        int y;

        MGB_TYPEINFO_OBJ_DECL;
    };

    MGB_TYPEINFO_OBJ_IMPL(Event0);
    MGB_TYPEINFO_OBJ_IMPL(Event1);

    struct TrackedHandle {
        static int nr_inst;

        TrackedHandle() {
            ++ nr_inst;
        }

        TrackedHandle(const TrackedHandle &) {
            ++ nr_inst;
        }

        ~TrackedHandle() {
            -- nr_inst;
        }

        void operator() (const Event0&) {
        }
    };
    int TrackedHandle::nr_inst = 0;
}

TEST(TestEvent, Simple) {
    SyncEventConnecter conn;

    int ev0_check = 0;
    auto on_ev0 = [&ev0_check](const Event0 &ev) {
        ASSERT_EQ(2, ev.x);
        ++ ev0_check;
    };

    int ev1_check = 0;
    auto on_ev1 = [&ev1_check](const Event1 &ev) {
        ASSERT_EQ(3, ev.y);
        ++ ev1_check;
    };

    conn.register_receiver_permanent<Event0>(on_ev0);
    auto ts = conn.version();
    conn.signal_inplace<Event0>(2);
    conn.signal_inplace<Event1>(3);

    ASSERT_EQ(ts, conn.version());

    {
        auto hdl = conn.register_receiver<Event1>(on_ev1);
        ASSERT_EQ(ts + 1, conn.version());
        conn.signal_inplace<Event0>(2);
        conn.signal_inplace<Event1>(3);
        ASSERT_EQ(ts + 1, conn.version());
    }
    ASSERT_EQ(ts + 2, conn.version());
    conn.signal_inplace<Event0>(2);
    conn.signal_inplace<Event1>(3);

    ASSERT_EQ(3, ev0_check);
    ASSERT_EQ(1, ev1_check);
}

TEST(TestEvent, MultiRecv) {
    SyncEventConnecter conn;

    int chk0 = 0, chk1 = 0, delta = 0;
    auto on_ev0 = [&delta](int *chk, const Event0 &ev) {
        ASSERT_EQ(2, ev.x);
        ++ delta;
        (*chk) += delta;
    };

    using namespace std::placeholders;

    auto hdl0 = conn.register_receiver<Event0>(std::bind(on_ev0, &chk0, _1)),
         hdl1 = conn.register_receiver<Event0>(std::bind(on_ev0, &chk1, _1));

    conn.signal_inplace<Event0>(2);
    ASSERT_EQ(1, chk0);
    ASSERT_EQ(2, chk1);

    hdl1.reset();
    conn.signal_inplace<Event0>(2);
    ASSERT_EQ(4, chk0);
    ASSERT_EQ(2, chk1);


    hdl0.reset();
    conn.signal_inplace<Event0>(2);
    ASSERT_EQ(4, chk0);
    ASSERT_EQ(2, chk1);
}

TEST(TestEvent, HandleDtor0) {
    ASSERT_EQ(0, TrackedHandle::nr_inst);
    SyncEventConnecter::ReceiverHandler hdl;
    SyncEventConnecter conn;
    hdl = conn.register_receiver<Event0>(TrackedHandle{});
    ASSERT_EQ(1, TrackedHandle::nr_inst);
    hdl.reset();
    ASSERT_EQ(0, TrackedHandle::nr_inst);
}

TEST(TestEvent, HandleDtor1) {
    ASSERT_EQ(0, TrackedHandle::nr_inst);
    SyncEventConnecter::ReceiverHandler hdl;
    {
        SyncEventConnecter conn;
        hdl = conn.register_receiver<Event0>(TrackedHandle{});

        ASSERT_EQ(1, TrackedHandle::nr_inst);
    }
    ASSERT_EQ(0, TrackedHandle::nr_inst);
    hdl.reset();
    ASSERT_EQ(0, TrackedHandle::nr_inst);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

