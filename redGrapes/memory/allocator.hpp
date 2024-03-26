/* Copyright 2023-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/memory/block.hpp"

#include <boost/core/demangle.hpp>
#include <spdlog/spdlog.h>

#include <memory>

namespace redGrapes
{
    namespace memory
    {
        struct Allocator
        {
            WorkerId worker_id;

            Allocator() : Allocator(*TaskFreeCtx::current_worker_id)
            {
            }

            // allocate on arena for specific worker
            Allocator(WorkerId worker_id) : worker_id(worker_id)
            {
            }

            Block allocate(size_t n_bytes)
            {
                return TaskFreeCtx::worker_alloc_pool->get_alloc(worker_id).allocate(n_bytes);
            }

            void deallocate(Block blk)
            {
                TaskFreeCtx::worker_alloc_pool->get_alloc(worker_id).deallocate(blk);
            }
        };

        template<typename T>
        struct StdAllocator
        {
            Allocator alloc;
            typedef T value_type;

            StdAllocator() : alloc()
            {
            }

            StdAllocator(WorkerId worker_id) : alloc(worker_id)
            {
            }

            template<typename U>
            constexpr StdAllocator(StdAllocator<U> const& other) noexcept : alloc(other.alloc)
            {
            }

            inline T* allocate(std::size_t n)
            {
                Block blk = alloc.allocate(sizeof(T) * n);
                SPDLOG_TRACE(
                    "allocate {},{},{}",
                    (uintptr_t) blk.ptr,
                    n * sizeof(T),
                    boost::core::demangle(typeid(T).name()));

                return (T*) blk.ptr;
            }

            inline void deallocate(T* p, std::size_t n = 0) noexcept
            {
                alloc.deallocate(Block{(uintptr_t) p, sizeof(T) * n});
            }

            template<typename U, typename... Args>
            void construct(U* p, Args&&... args)
            {
                new(p) U(std::forward<Args>(args)...);
            }

            template<typename U>
            void destroy(U* p)
            {
                p->~U();
            }
        };

        template<typename T, typename U>
        bool operator==(StdAllocator<T> const&, StdAllocator<U> const&)
        {
            return true;
        }

        template<typename T, typename U>
        bool operator!=(StdAllocator<T> const&, StdAllocator<U> const&)
        {
            return false;
        }

        /* allocates a shared_ptr in the memory pool of a given worker
         */
        template<typename T, typename... Args>
        std::shared_ptr<T> alloc_shared_bind(WorkerId worker_id, Args&&... args)
        {
            return std::allocate_shared<T>(StdAllocator<T>(worker_id), std::forward<Args>(args)...);
        }

        /* allocates a shared_ptr in the memory pool of the current worker
         */
        template<typename T, typename... Args>
        std::shared_ptr<T> alloc_shared(Args&&... args)
        {
            return std::allocate_shared<T>(StdAllocator<T>(), std::forward<Args>(args)...);
        }

    } // namespace memory
} // namespace redGrapes
