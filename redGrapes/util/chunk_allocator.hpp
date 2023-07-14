/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <redGrapes/util/spinlock.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/dispatch/thread/local.hpp>
#include <redGrapes/dispatch/thread/cpuset.hpp>

namespace redGrapes
{
namespace memory
{

struct Chunk
{
    Chunk( size_t capacity );
    Chunk( Chunk const & ) = delete;
    Chunk( Chunk & ) = delete;
    ~Chunk();
    
    bool empty() const;
    void reset();

    void * m_alloc( size_t n_bytes );
    unsigned m_free( void * );

    bool contains( void * ) const;

private:
    std::atomic_ptrdiff_t offset;
    size_t capacity;
    uintptr_t base;

    std::atomic<unsigned> count;

    // TODO: linked list of blocked chunks
    //    ---> avoids std::vector in ChunkAllocator
    // std::optional< std::unique_ptr< Chunk > > prev;
};

struct ChunkAllocator
{
    SpinLock m;
    size_t chunk_size;
    std::vector< std::unique_ptr<Chunk> > blocked_chunks;
    std::unique_ptr<Chunk> active_chunk;

    ChunkAllocator( size_t chunk_size )
        : active_chunk( std::make_unique<Chunk>(chunk_size) )
        , chunk_size( chunk_size )
    {
        blocked_chunks.reserve(64);
    }

    template <typename T>
    T * allocate( std::size_t n = 1 )
    {
        size_t s = n * sizeof(T);
        s--;
        s |= s >> 0x1;
        s |= s >> 0x2;
        s |= s >> 0x4;
        s |= s >> 0x8;
        s |= s >> 0x10;
        s |= s >> 0x20;
        s++;

        T * item = (T*) active_chunk->m_alloc( s );

        if( !item )
        {
            // create new chunk & try again
            {
                std::lock_guard<SpinLock> lock(m);
                blocked_chunks.emplace_back(std::make_unique<Chunk>( chunk_size ));
                std::swap(active_chunk, blocked_chunks[blocked_chunks.size()-1]);
            }
            item = (T*) active_chunk->m_alloc( s );
        }

        return item;
    }

    template <typename T>
    void deallocate( T * ptr )
    {
        if( active_chunk->contains((void*)ptr) )
            active_chunk->m_free((void*)ptr);

        else
        {
            std::lock_guard<SpinLock> lock(m);

            // find chunk containing ptr
            for( unsigned i = 0; i < blocked_chunks.size(); ++i )
            {
                Chunk & c = *blocked_chunks[i];

                if( c.contains((void*)ptr) )
                {
                    if( c.m_free( (void*) ptr ) == 0 )
                        blocked_chunks.erase(std::begin(blocked_chunks) + i);

                    break;
                }
            }
        }
    }
};


struct NUMAChunkAllocator
{
    std::vector< std::unique_ptr<ChunkAllocator> > arenas;

    NUMAChunkAllocator( size_t chunk_size )
    {
        arenas.reserve(65);
        for(unsigned i =0; i < 65; ++i)
        {
            if( i > 0)
                dispatch::thread::pin_cpu( i - 1 );

            arenas.push_back( std::make_unique<ChunkAllocator>(chunk_size) );
        }

        dispatch::thread::unpin_cpu();
    }

    template <typename T>
    T * allocate( std::size_t n = 1 )
    {
        unsigned numa_domain = (redGrapes::dispatch::thread::current_waker_id);
        T * ptr = arenas[numa_domain]->template allocate<T>( n );
        return ptr;
    }

    template <typename T>
    void deallocate( T * ptr )
    {
        unsigned numa_domain = (redGrapes::dispatch::thread::current_waker_id);
        arenas[numa_domain]->template deallocate<T>( ptr );
    }    
};


} // namespace memory
} // namespace redGrapes

