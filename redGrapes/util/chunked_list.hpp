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

#include <mutex>
#include <shared_mutex>

namespace redGrapes
{
    // A lock-free, append/remove container
template < typename T, size_t chunk_size = 1024 >
    struct ChunkedList
    {
        struct Chunk
        {
            unsigned id;

            std::atomic< unsigned > next_id;
            std::array< T*, chunk_size > buf;
            std::shared_ptr< Chunk > next;

            Chunk( unsigned id )
                : id(id)
                , next_id(0)
            {}
        };

        std::shared_ptr< Chunk > head;
        std::atomic_int next_chunk_id;

        ChunkedList()
            : head( std::make_shared<Chunk>(0) ),
              next_chunk_id(1)
        {
        }
        
        unsigned push(T * item)
        {
        retry:
            auto chunk = head;

            unsigned id = chunk->next_id.fetch_add(1);
            if( id < chunk_size )
            {
                chunk->buf[id] = item;
                return chunk->id * chunk_size + id;
            }
            else if( id == chunk_size )
            {
                // create new chunk
                auto new_chunk = std::make_shared< Chunk >(next_chunk_id.fetch_add(1));
                new_chunk->next = head;
                head = new_chunk;

                SPDLOG_TRACE("new chunk");

                goto retry;
            }
            else
            {                
                // wait for new chunk
                while( head != chunk->next );

                // try again
                goto retry;
            }
        }
        
        void remove(unsigned idx)
        {            
            auto chunk = head;
            while( chunk != nullptr )
            {
                if( chunk->id == idx / chunk_size )
                {
                    chunk->buf[ idx % chunk_size ] = nullptr;
                    return;
                }
                chunk = chunk->next;
            }

            // out ouf range
            return;
        }

        /* ITERATOR */
        /************/

        struct BackwardsIterator
        {
            std::shared_ptr< Chunk > chunk;
            int idx;

            bool operator!=(BackwardsIterator const & other)
            {
                return this->chunk != other.chunk;// || this->idx != other.idx;
            }

            BackwardsIterator& operator++()
            {
                while( chunk )
                {
                    --idx;
                    if( idx < 0 )
                    {
                        idx = chunk_size-1;
                        chunk = chunk->next;
                        if( !chunk )
                            break;
                    }

                    if( chunk->buf[idx] != nullptr )
                        break;
                }
                
                return *this;
            }

            T * operator* ()
            {
                return chunk->buf[idx];
            }
        };

        std::pair<BackwardsIterator, BackwardsIterator> iter_from(unsigned idx)
        {
            SPDLOG_TRACE("iter from {}", idx);

            auto chunk = head;
            while( chunk != nullptr )
            {
                if( chunk->id == idx / chunk_size )
                {
                    return std::make_pair(
                               BackwardsIterator{ chunk, idx%chunk_size },
                               BackwardsIterator{ nullptr, chunk_size-1 }
                           );
                }
                chunk = chunk->next;
            }

            return std::make_pair(
                               BackwardsIterator{ nullptr, chunk_size-1 },
                               BackwardsIterator{ nullptr, chunk_size-1 }
                           );
        }

    };

} // namespace redGrapes

