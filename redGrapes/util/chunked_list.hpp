/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/util/chunked_list.hpp
 */

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <redGrapes/util/allocator.hpp>

namespace redGrapes
{
    // A lock-free, append/remove container
    template<typename T, size_t chunk_size = 1024>
    struct ChunkedList
    {
        struct Chunk
        {
            unsigned chunk_id;

            std::atomic<unsigned> next_item_id;
            std::array< std::optional<T>, chunk_size > buf;
            std::shared_ptr< Chunk > next;

            Chunk(unsigned id) : chunk_id(id), next_item_id(0)
            {
            }
        };

        std::atomic_int next_chunk_id;
        std::shared_ptr<Chunk> head;

        ChunkedList() : next_chunk_id(0)
        {
        }

        ChunkedList(ChunkedList const& other) : next_chunk_id(0)
        {
            for(auto& e : other)
                push(e);
        }

        unsigned push(T const& item)
        {
        retry:
            unsigned id = chunk_size;
            std::shared_ptr<Chunk> chunk = head;

            if(chunk)
                id = chunk->next_item_id.fetch_add(1);

            if(id < chunk_size)
            {
                chunk->buf[id] = item;
                return chunk->chunk_id * chunk_size + id;
            }
            else if(id == chunk_size)
            {
                // create new chunk
                auto new_chunk = memory::alloc_shared<Chunk>( next_chunk_id.fetch_add(1) );
                new_chunk->next = head;

                // only set head if no other thread created a new chunk
                // head.compare_exchange_strong( chunk, new_chunk );
                head = new_chunk;

                goto retry;
            }
            else
            {
                // wait for new chunk
                while(head != chunk->next)
                    ;

                // try again
                goto retry;
            }
        }

        void remove(unsigned idx)
        {
            std::shared_ptr<Chunk> chunk = head;
            while(chunk != nullptr)
            {
                if(chunk->chunk_id == idx / chunk_size)
                {
                    chunk->buf[idx % chunk_size] = std::nullopt;
                    return;
                }
                chunk = chunk->next;
            }

            // out ouf range
            throw std::out_of_range("");
            return;
        }

        void erase(T item)
        {
            std::shared_ptr<Chunk> chunk = head;
            while(chunk != nullptr)
            {
                for(unsigned idx = 0; idx < chunk_size; ++idx)
                    if(chunk->buf[idx] == item)
                    {
                        chunk->buf[idx] = std::nullopt;
                        // return;
                    }

                chunk = chunk->next;
            }
        }

        /* ITERATOR */
        /************/

        struct BackwardsIterator
        {
            std::shared_ptr<Chunk> chunk;
            int idx;

            bool operator!=(BackwardsIterator const& other)
            {
                return this->chunk != other.chunk; // || this->idx != other.idx;
            }

            BackwardsIterator& operator++()
            {
                while(chunk)
                {
                    --idx;
                    if(idx < 0)
                    {
                        idx = chunk_size - 1;
                        chunk = chunk->next;
                        if(!chunk)
                            break;
                    }

                    if(is_some())
                        break;
                }

                return *this;
            }

            bool is_some() const
            {
                return (bool) (chunk->buf[idx]);
            }

            T& operator*() const
            {
                return *chunk->buf[idx];
            }
        };

        auto begin_from(unsigned idx) const
        {
            std::shared_ptr<Chunk> chunk = head;
            while(chunk != nullptr)
            {
                if(chunk->chunk_id == idx / chunk_size)
                {
                    auto s = BackwardsIterator{chunk, (int) (idx % chunk_size)};
                    if(!s.is_some())
                        ++s;

                    return s;
                }
                chunk = chunk->next;
            }

            return end();
        }

        auto begin() const
        {
            unsigned len = 0;
            if(head)
                len = (next_chunk_id - 1) * chunk_size + (head->next_item_id - 1);

            return begin_from(len);
        }

        auto end() const
        {
            return BackwardsIterator{nullptr, chunk_size - 1};
        }

        auto iter() const
        {
            return std::make_pair(begin(), end());
        }

        std::pair<BackwardsIterator, BackwardsIterator> iter_from(unsigned idx) const
        {
            return std::make_pair(begin_from(idx), end());
        }
    };

} // namespace redGrapes
