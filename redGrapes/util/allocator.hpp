#pragma once

#include <redGrapes/util/chunk_allocator.hpp>

namespace redGrapes
{
namespace memory
{

template < typename T >
struct Allocator
{
    typedef T value_type;
 
    Allocator () = default;

    template< typename U >
    constexpr Allocator(Allocator<U> const&) noexcept {}

    static ChunkAllocator<> & get_instance()
    {
        static ChunkAllocator<> chunkalloc;
        return chunkalloc;
    }
    
    T* allocate( std::size_t n )
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        T * p = get_instance().allocate< T >( n );
        if( p )
            return p;
        else
            throw std::bad_alloc();
    }
 
    void deallocate(T* p, std::size_t n) noexcept
    {
        get_instance().deallocate< T >( p );
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
std::shared_ptr<T> alloc_shared( Args&&... args )
{
    return std::allocate_shared< T, Allocator<T> >( Allocator<T>(), std::forward<Args>(args)... );
}

} // namespace memory
} // namespace redGrapes


