/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>

#include <hwloc.h>
#include <spdlog/spdlog.h>

namespace redGrapes
{
namespace memory
{

/* maintains a lockfree singly-linked list of arbitrarily sized data
 * chunks
 *
 * allowed operations:
 *   - append new chunks at head
 *   - erase any chunk which is not current head
 *   - reversed iteration (starting at head)
 *
 * each chunk is managed through a `std::shared_ptr` which points to a
 * contiguous block containing list-metadata, the chunk-control-object
 * (`ChunkData`) and freely usable data.
 */
template <
    typename ChunkData,
    template <typename> typename Allocator
>
struct ChunkList
{
private:
    struct Chunk
    {
        bool volatile deleted;
        std::shared_ptr< Chunk > prev;

        template < typename... Args >
        Chunk( Args&&... args ) : deleted(false), prev(nullptr)
        {
            new ( get() ) ChunkData ( std::forward<Args>(args)... );
        }

        ~Chunk()
        {
            if( !deleted )
                spdlog::error("dropping chunk which was never deleted");

            assert( deleted );

            get()->~ChunkData();
        }

        /* flag this chunk as deleted and call ChunkData destructor
         */
        void erase()
        {
            deleted = true;
        }

        /* adjusts `prev` so that it points to a non-deleted chunk again
         * this should free the shared_ptr of the original prev
         * in case no iterators point to it
         */
        void skip_deleted_prev()
        {
            while( prev && prev->deleted )
                prev = prev->prev;
        }

        ChunkData * get() const
        {
            if( !deleted )
                return (ChunkData*)((uintptr_t)this + sizeof(Chunk));
            else
                throw std::runtime_error("ChunkList: get() on deleted chunk");
        }
    };

private:
    Allocator< uint8_t > alloc;
    std::shared_ptr< Chunk > head;
    size_t const chunk_size;

public:
    ChunkList( Allocator< uint8_t > alloc, size_t chunk_size )
        : alloc( alloc )
        , head( nullptr )
        , chunk_size( chunk_size )
    {}

    /* initializes a new chunk
     */
    template < typename... Args >
    void add_chunk( Args&&... args )
    {
        size_t const alloc_size = sizeof(Chunk) + chunk_size;

        Chunk * c = (Chunk *) alloc.allocate( alloc_size );

        new ( c ) Chunk ( std::forward<Args>(args)... );

        std::shared_ptr< Chunk > new_chunk(
            c,
            [this, alloc_size]( Chunk * c ){
                this->alloc.deallocate( (uint8_t*)c, alloc_size );
            },
            alloc 
        );

        append_chunk( new_chunk );
    }

    /* atomically appends a floating chunk to this list
     */
    void append_chunk( std::shared_ptr< Chunk > new_head )
    {
        bool append_successful = false;
        while( ! append_successful )
        {
            std::shared_ptr< Chunk > old_head = head;
            new_head->prev = old_head;
            append_successful = std::atomic_compare_exchange_strong<Chunk>( &head, &old_head, new_head );
        }
    }

    template < bool is_const = false >
    struct BackwardIterator
    {
        std::shared_ptr< Chunk > c;

        void erase()
        {
            c->erase();
        }

        bool operator!=(BackwardIterator const & other) const
        {
            return c != other.c;
        }

        operator bool() const
        {
            return (bool)c;
        }

        typename std::conditional<
            is_const,
            ChunkData const *,
            ChunkData *
        >::type
        operator->() const
        {
            return c->get();
        }
        
        typename std::conditional<
            is_const,
            ChunkData const &,
            ChunkData &
        >::type
        operator*() const
        {
            return *c->get();
        }
        
        BackwardIterator& operator++()
        {
            if( c )
            {
                c->skip_deleted_prev();
                c = c->prev;
            }

            return *this;
        }
    };

    using ConstBackwardIterator = BackwardIterator< true >;
    using MutBackwardIterator = BackwardIterator< false >;

    /* get iterator starting at current head, iterating backwards from
     * most recently added to least recently added
     */
    MutBackwardIterator rbegin() const
    {
        return MutBackwardIterator{ head };
    }

    MutBackwardIterator rend() const
    {
        return MutBackwardIterator{ std::shared_ptr<Chunk>() };
    }

    ConstBackwardIterator crbegin() const
    {
        return ConstBackwardIterator{ head };
    }

    ConstBackwardIterator crend() const
    {
        return ConstBackwardIterator{ std::shared_ptr<Chunk>() };
    }
    
    /* Flags chunk at `pos` as erased. Actual removal is delayed until
     * iterator stumbles over it.
     *
     * Since we only append to the end and `chunk` is not `head`,
     * there wont occur any inserts after this chunk.
     */
    void erase( MutBackwardIterator pos )
    {
        pos.erase();
    }

};

} // namespace memory

} // namespace redGrapes

