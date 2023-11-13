/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <memory>
#include <optional>

#include <hwloc.h>
#include <spdlog/spdlog.h>

namespace redGrapes
{
namespace memory
{

/* maintains a lockfree singly-linked list
 * with the following allowed operations:
 *   - append new chunks at head
 *   - erase any chunk which is not current head
 *   - reversed iteration (starting at head)
 *
 * each chunk is managed through a `std::shared_ptr` which points to a
 * contiguous block containing list-metadata, the chunk-control-object
 * (`ChunkData`) and freely usable data.
 *
 * @tparam Item element type
 * @tparam Allocator must satisfy `Allocator` concept
 */
template <
    typename Item,
    template <typename> class Allocator
>
struct AtomicList
{
//private:
    struct ItemPtr
    {
        bool volatile deleted;
        std::shared_ptr< ItemPtr > prev;
        Item * item_data;

        template < typename... Args >
        ItemPtr( Item * item_data, Args&&... args )
                : deleted(false)
                , prev(nullptr)
                , item_data(item_data)
        {
            new ( get() ) Item ( std::forward<Args>(args)... );
        }

        ~ItemPtr()
        {
            SPDLOG_INFO("destruct chunk {}", (void*)item_data);
            get()->~Item();
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
            std::shared_ptr<ItemPtr> p = std::atomic_load( &prev );
            while( p && p->deleted )
                p = std::atomic_load( &p->prev );

            std::atomic_store( &prev, p );
        }

        Item * get() const
        {
            return item_data;
        }
    };

    Allocator< uint8_t > alloc;
    std::shared_ptr< ItemPtr > head;
    size_t const chunk_size;

    /* keeps a single, predefined pointer
     * and frees it on deallocate.
     * used to spoof the allocated size to be bigger than requested.
     */
    template <typename T>
    struct StaticAlloc
    {
        typedef T value_type;

        Allocator< T > alloc;
        T * ptr;

        StaticAlloc( Allocator<uint8_t> alloc, size_t n_bytes )
            : alloc(alloc)
            , ptr( (T*)alloc.allocate( n_bytes ) )
        {}

        template<typename U>
        constexpr StaticAlloc( StaticAlloc<U> const & other ) noexcept
            : alloc(other.alloc)
            , ptr((T*)other.ptr)
        {}

        T * allocate( size_t n ) noexcept
        {
            return ptr;
        }

        void deallocate( T * _p, std::size_t _n ) noexcept
        {
            alloc.deallocate( ptr, 1 );
        }
    };



public:
            
    AtomicList( Allocator< uint8_t > && alloc, size_t chunk_size )
        : alloc( alloc )
        , head( nullptr )
        , chunk_size( chunk_size )
    {
#ifndef NDEBUG
        if( chunk_size <= get_controlblock_size() )
             spdlog::error("chunksize = {}, control block ={}", chunk_size, get_controlblock_size());
#endif

        assert( chunk_size > get_controlblock_size() );
    }

    static constexpr size_t get_controlblock_size()
    {
        /* TODO: use sizeof( ...shared_ptr_inplace_something... )
         */
        size_t const shared_ptr_size = 512;

        return sizeof(ItemPtr) + shared_ptr_size;
    }

    constexpr size_t get_chunk_capacity()
    {
        return chunk_size - get_controlblock_size();
    }

    /* initializes a new chunk
     */
    void allocate_item()
    {
        TRACE_EVENT("Allocator", "AtomicList::allocate_item()");

        /* NOTE: we are relying on std::allocate_shared
         * to do one *single* allocation which contains:
         * - shared_ptr control block
         * - chunk control block
         * - chunk data
         * whereby chunk data is not included by sizeof(Chunk),
         * but reserved by StaticAlloc.
         * This works because shared_ptr control block lies at lower address.
         */
        StaticAlloc<void> chunk_alloc( this->alloc, chunk_size );
        uintptr_t base = (uintptr_t)chunk_alloc.ptr;
        append_item(
            std::allocate_shared< ItemPtr >(
                chunk_alloc,

                /* TODO: generalize this constructor call,
                 *  specialized for `memory chunks` now
                 */
                (Item*) (base + get_controlblock_size()),
                base + get_controlblock_size() + sizeof(Item),
                base + chunk_size
            )
        );
    }

    /* atomically appends a floating chunk to this list
     */
    void append_item( std::shared_ptr< ItemPtr > new_head )
    {
        TRACE_EVENT("Allocator", "AtomicList::append_item()");
        bool append_successful = false;
        while( ! append_successful )
        {
            std::shared_ptr< ItemPtr > old_head = std::atomic_load( &head );
            std::atomic_store( &new_head->prev, old_head );
            append_successful = std::atomic_compare_exchange_strong<ItemPtr>( &head, &old_head, new_head );
        }
    }

    template < bool is_const = false >
    struct BackwardIterator
    {
        std::shared_ptr< ItemPtr > c;

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
            Item const *,
            Item *
        >::type
        operator->() const
        {
            return c->get();
        }
        
        typename std::conditional<
            is_const,
            Item const &,
            Item &
        >::type
        operator*() const
        {
            return *c->get();
        }

        void optimize()
        {
            if(c)
                c->skip_deleted_prev();
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
        return MutBackwardIterator{ std::atomic_load(&head) };
    }

    MutBackwardIterator rend() const
    {
        return MutBackwardIterator{ std::shared_ptr<ItemPtr>() };
    }

    ConstBackwardIterator crbegin() const
    {
        return ConstBackwardIterator{ std::atomic_load(&head) };
    }

    ConstBackwardIterator crend() const
    {
        return ConstBackwardIterator{ std::shared_ptr<ItemPtr>() };
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

