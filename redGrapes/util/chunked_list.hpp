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

#include <cassert>
#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <redGrapes/util/allocator.hpp>
#include <spdlog/spdlog.h>

namespace redGrapes
{

    // A lock-free, append/remove container
    template<typename T, size_t T_chunk_size = 1024>
    struct ChunkedList
    {
    private:
        using Chunk = std::optional<T>*;
        size_t const chunk_size;
        Chunk * chunks;
        std::atomic< uint16_t > chunks_capacity;
        std::atomic< uint16_t > chunks_count;
        std::atomic< uint32_t > next_item_id;

        void init_chunk( )
        {
            unsigned chunk_idx = chunks_count.fetch_add(1);

            if( chunk_idx >= chunks_capacity )
                resize_superchunk( chunks_capacity * 2 );
            
            std::optional<T>* new_chunk = memory::Allocator<std::optional<T>>().allocate(chunk_size);
            for( int i = 0; i < chunk_size; ++i )
                new_chunk[i] = std::nullopt;

            chunks[ chunk_idx ] = new_chunk;
        }

        void resize_superchunk( size_t new_chunks_capacity )
        {
            assert( new_chunks_capacity > chunks_capacity );

            spdlog::info("resize to {} chunks", new_chunks_capacity);

            Chunk * new_chunks = memory::Allocator< Chunk >().allocate( new_chunks_capacity );

            for( int i = 0; i < chunks_capacity; ++i )
                new_chunks[i] = chunks[i];
            for( int i = chunks_capacity; i < new_chunks_capacity; ++i )
                new_chunks[i] = nullptr;

            auto old_chunks = chunks;
            chunks = new_chunks;

            memory::Allocator< Chunk >().deallocate( old_chunks, chunks_capacity );
            chunks_capacity = new_chunks_capacity;
        }

    public:
        ~ChunkedList()
        {
            for( unsigned i = 0; i < chunks_count; ++i )
                if( chunks[i] )
                    memory::Allocator< std::optional<T> >().deallocate( chunks[i], chunk_size );
            memory::Allocator< Chunk >().deallocate( chunks, chunks_capacity );
        }

        ChunkedList( size_t chunk_size = T_chunk_size )
            : chunk_size( chunk_size )
            , next_item_id( 0 )
            , chunks_count( 0 )
            , chunks_capacity( 16 )
        {
            chunks = memory::Allocator< Chunk >().allocate( chunks_capacity );
            for( int i = 0; i < chunks_capacity; ++i )
                chunks[i] = nullptr;
        }

        ChunkedList( ChunkedList && other ) = default;
        
        ChunkedList( ChunkedList const& other )
            : ChunkedList()
        {
            SPDLOG_TRACE("copy construct ChunkedList!!");
            reserve( other.size() );
            for( auto& e : other )
                push(e);
        }

        unsigned size() const noexcept
        {
            return next_item_id;
        }

        unsigned capacity() const noexcept
        {
            return chunks_count * chunk_size;
        }

        unsigned free_capacity() const noexcept
        {
            return capacity() - next_item_id;
        }

        void reserve( size_t required_size )
        {
            if( required_size > free_capacity() ) {
                reserve_chunks(
                    1 +
                    ((required_size - free_capacity() - 1) / chunk_size)
                );
            }
        }

        void reserve_chunks( size_t n )
        {
            SPDLOG_TRACE("reserve {} chunks", n);

            if( chunks_count + n > chunks_capacity )
                resize_superchunk( chunks_count + n );

            assert( chunks_count + n <= chunks_capacity );

            for( size_t i = 0; i < n; ++i )
                init_chunk();
        }

        std::optional< T > & get( unsigned idx )
        {
            unsigned chunk_idx = idx / chunk_size;
            unsigned chunk_off = idx % chunk_size;

            assert( chunk_idx < chunks_count );

            if( chunks[ chunk_idx ] )
                return chunks[ chunk_idx ][ chunk_off ];
            else
                throw std::runtime_error("invalid index: missing chunk");
        }

        std::optional< T > const & get( unsigned idx ) const
        {
            unsigned chunk_idx = idx / chunk_size;
            unsigned chunk_off = idx % chunk_size;

            assert( chunk_idx < chunks_count );

            if( chunks[ chunk_idx ] )
                return chunks[ chunk_idx ][ chunk_off ];
            else
                throw std::runtime_error("invalid index: missing chunk");
        }

        unsigned push( T const& item )
        {            
            unsigned idx = next_item_id.fetch_add(1);
            unsigned chunk_idx = idx / chunk_size;
            unsigned chunk_off = idx % chunk_size;

            while( chunk_idx >= chunks_capacity
                || chunks[ chunk_idx ] == nullptr )
                init_chunk();

            chunks[ chunk_idx ][ chunk_off ] = item;

            return idx;
        }

        void remove(unsigned idx)
        {
            assert( idx < size() );
            get( idx ) = std::nullopt;
        }

        void erase( T item )
        {
            for( unsigned i = 0; i < size(); ++i )
            {
                auto & x = get(i);
                if( x )
                {
                    if( *x == item )
                        remove( i );
                }
            }
        }

        /* ITERATOR */
        /************/
        struct ConstIterator
        {
            ChunkedList< T, T_chunk_size > const & c;
            int idx;

            bool operator!=(ConstIterator const& other)
            {
                return this->idx != other.idx;
            }

            ConstIterator& operator--()
            {
                do {
                    --idx;
                    if( idx < 0 )
                        return *this;
                }
                while( !is_some() );

                return *this;
            }

            ConstIterator& operator++()
            {
                do {
                    ++idx;
                    if( idx >= c.size() )
                    {
                        idx = -1;
                        return *this;
                    }
                }
                while( !is_some() );

                return *this;
            }

            bool is_some() const
            {
                if( idx < c.size() && idx >= 0 )
                    return (bool) (c.get(idx));
                else
                    return false;
            }

            T const & operator*() const
            {
                assert( is_some() );
                return *c.get(idx);
            }            
        };

        struct Iterator
        {
            ChunkedList< T, T_chunk_size > & c;
            int idx;

            bool operator!=(Iterator const& other)
            {
                return this->idx != other.idx;
            }

            Iterator& operator--()
            {
                do {
                    --idx;
                    if( idx < 0 )
                        return *this;
                }
                while( !is_some() );

                return *this;
            }

            Iterator& operator++()
            {
                do {
                    ++idx;
                    if( idx >= c.size() )
                    {
                        idx = -1;
                        return *this;
                    }
                }
                while( !is_some() );

                return *this;
            }

            bool is_some() const
            {
                if( idx < c.size() && idx >= 0 )
                    return (bool) (c.get(idx));
                else
                    return false;
            }

            T& operator*() const
            {
                assert( is_some() );
                return *c.get(idx);
            }
        };

        auto begin_from(unsigned idx)
        {
            auto it = Iterator{ *this, (int)idx };

            if( ! it.is_some() )
                --it;

            return it;
        }

        auto begin()
        {
            return begin_from(0);
        }

        auto end()
        {
            return Iterator{ *this, -1 };
        }

        auto begin_from(unsigned idx) const
        {
            auto it = ConstIterator{ *this, (int)idx };

            if( ! it.is_some() )
                --it;

            return it;
        }
        
        auto begin() const
        {
            return begin_from(0);
        }

        auto end() const
        {
            return ConstIterator{ *this, -1 };
        }

        auto iter()
        {
            return std::make_pair(begin(), end());
        }

        std::pair<Iterator, Iterator> iter_from(unsigned idx)
        {
            return std::make_pair(begin_from(idx), end());
        }
    };

} // namespace redGrapes

