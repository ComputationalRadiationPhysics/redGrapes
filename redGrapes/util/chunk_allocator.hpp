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
    alignas(64) std::atomic_ptrdiff_t offset;
    size_t capacity;
    uintptr_t base;

    alignas(64) std::atomic<unsigned> count;
};

template < size_t chunk_size = 0x800000 >
struct ChunkAllocator
{
    SpinLock m;
    std::vector< std::unique_ptr<Chunk> > blocked_chunks;
    std::unique_ptr<Chunk> active_chunk;

    ChunkAllocator()
        : active_chunk( std::make_unique<Chunk>(chunk_size) )
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

} // namespace memory
} // namespace redGrapes

