/**
 * \file imperative/src/impl/async_releaser.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node.h"
#include "megbrain/imperative/blob_manager.h"
#include "megbrain/system.h"

#include "./event_pool.h"

namespace mgb {
namespace imperative {

class AsyncReleaser : public CompNodeDepedentObject {
    struct WaiterParam {
        CompNode cn;
        CompNode::Event* event;
        BlobPtr blob;
        HostTensorStorage::RawStorage storage;
    };
    class Waiter final : public AsyncQueueSC<WaiterParam, Waiter> {
        AsyncReleaser* m_par_releaser;

    public:
        // disable busy wait by set max_spin=0 to save CPU cycle
        Waiter(AsyncReleaser* releaser)
                : AsyncQueueSC<WaiterParam, Waiter>(0),
                  m_par_releaser(releaser) {}

        void process_one_task(WaiterParam& param) {
            if (param.event->finished()) {
                param.blob.reset();
                param.storage.reset();
                EventPool::without_timer().free(param.event);
                return;
            }

            using namespace std::literals;
            std::this_thread::sleep_for(1us);
            add_task(std::move(param));
        }
        void on_async_queue_worker_thread_start() override {
            sys::set_thread_name("releaser");
        }
    };
    Waiter m_waiter{this};

protected:
    std::shared_ptr<void> on_comp_node_finalize() override {
        m_waiter.wait_task_queue_empty();
        return {};
    }

public:
    static AsyncReleaser* inst() {
        static AsyncReleaser releaser;
        return &releaser;
    }

    ~AsyncReleaser() {
        m_waiter.wait_task_queue_empty();
    }

    void add(BlobPtr blob, CompNode cn) { add(cn, std::move(blob), {}); }

    void add(const HostTensorND& hv) {
        add(hv.comp_node(), {}, hv.storage().raw_storage());
    }

    void add(CompNode cn, BlobPtr blob,
             HostTensorStorage::RawStorage storage = {}) {
        auto event = EventPool::without_timer().alloc(cn);
        event->record();
        m_waiter.add_task({cn, event, std::move(blob), std::move(storage)});
    }
};
}
}
