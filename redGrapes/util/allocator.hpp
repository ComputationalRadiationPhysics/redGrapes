#pragma once

#include <redGrapes/util/multi_arena_alloc.hpp>

namespace redGrapes
{
namespace memory
{

extern std::shared_ptr< MultiArenaAlloc > alloc;
extern thread_local unsigned current_arena;

template < typename T >
struct Allocator
{
    unsigned arena_id;

    typedef T value_type;

    Allocator () : arena_id( current_arena )
    {
    }
    Allocator( unsigned arena_id ) : arena_id( arena_id ) {}

    template< typename U >
    constexpr Allocator(Allocator<U> const& other) noexcept
        : arena_id( other.arena_id )
    {
    }

    T* allocate( std::size_t n )
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        T * p = alloc->allocate< T >( arena_id, n );
        if( p )
            return p;
        else
            throw std::bad_alloc();
    }
 
    void deallocate(T* p, std::size_t n) noexcept
    {
        alloc->deallocate< T >( arena_id, p );
    }

    template < typename U, typename... Args >
    void construct(U * p, Args&&... args )
    {
        new (p) U ( std::forward<Args>(args)... );
    }

    template < typename U >
    void destroy( U * p )
    {
        p->~U();
    }
};

template<typename T, typename U>
bool operator==(Allocator<T> const&, Allocator<U> const &) { return true; }
 
template<typename T, typename U>
bool operator!=(Allocator<T> const&, Allocator<U> const&) { return false; }

template < typename T, typename... Args >
std::shared_ptr<T> alloc_shared_bind( unsigned arena_id, Args&&... args )
{
    return std::allocate_shared< T, Allocator<T> >( Allocator<T>( arena_id ), std::forward<Args>(args)... );
}

template < typename T, typename... Args >
std::shared_ptr<T> alloc_shared( Args&&... args )
{
    return std::allocate_shared< T, Allocator<T> >( Allocator<T>(), std::forward<Args>(args)... );
}

} // namespace memory
} // namespace redGrapes


