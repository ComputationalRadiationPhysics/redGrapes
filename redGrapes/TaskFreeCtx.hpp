/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/memory/chunked_bump_alloc.hpp"
#include "redGrapes/memory/hwloc_alloc.hpp"
#include "redGrapes/sync/cv.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace redGrapes
{

    using WorkerId = uint8_t;
    using ResourceId = uint16_t;

    /** WorkerID of parser to wake it up
     * ID 0,1,2... are used for worker threads
     * ID -1 is used as the default value for WorkerId, indicating uninitialized/no workers
     * Using the magic number -2 for the parser thread
     */
    constexpr WorkerId parserID = -2;

    // seperated to not templatize allocators with Task type
    struct WorkerAllocPool
    {
    public:
        inline memory::ChunkedBumpAlloc<memory::HwlocAlloc>& get_alloc(WorkerId worker_id)
        {
            assert(worker_id < allocs.size());
            return *allocs[worker_id];
        }

        std::vector<std::unique_ptr<memory::ChunkedBumpAlloc<memory::HwlocAlloc>>> allocs;
    };

    struct TaskFreeCtx
    {
        static inline HwlocContext hwloc_ctx;
        static inline WorkerId n_pus{
            static_cast<WorkerId>(hwloc_get_nbobjs_by_type(hwloc_ctx.topology, HWLOC_OBJ_PU))};
        static inline WorkerId n_workers;
        static inline WorkerAllocPool worker_alloc_pool;
        static inline CondVar cv{0};

        static inline ResourceId create_resource_uid()
        {
            static std::atomic<ResourceId> id = 0;
            return id++;
        }

        static inline std::function<void()> idle = []
        {
            SPDLOG_TRACE("Parser::idle()");

            /* the main thread shall not do any busy waiting
             * and always sleep right away in order to
             * not block any worker threads (those however should
             * busy-wait to improve latency)
             */
            cv.wait();
        };
        static inline thread_local std::optional<WorkerId> current_worker_id;
    };
} // namespace redGrapes
