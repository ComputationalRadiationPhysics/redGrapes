/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/memory/chunked_bump_alloc.hpp"
#include "redGrapes/memory/hwloc_alloc.hpp"

namespace redGrapes
{

    using WorkerId = unsigned;

    // seperated to not templatize allocators with Task type
    struct WorkerAllocPool
    {
    public:
        inline memory::ChunkedBumpAlloc<memory::HwlocAlloc>& get_alloc(WorkerId worker_id)
        {
            assert(worker_id < allocs.size());
            return allocs[worker_id];
        }

        std::vector<memory::ChunkedBumpAlloc<memory::HwlocAlloc>> allocs;
    };

    struct TaskFreeCtx
    {
        static inline unsigned n_workers;
        static inline unsigned n_pus;
        static inline HwlocContext hwloc_ctx;
        static inline std::shared_ptr<WorkerAllocPool> worker_alloc_pool;

        static inline thread_local std::function<void()> idle = [] {};
        static inline thread_local std::optional<WorkerId> current_worker_id;
    };
} // namespace redGrapes
