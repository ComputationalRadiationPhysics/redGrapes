#pragma once

#include <boost/core/demangle.hpp>
#include <memory>
#include <redGrapes/dispatch/thread/local.hpp>
// #include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/memory/block.hpp>
#include <redGrapes/resource/access/area.hpp>
//#include <redGrapes/context.hpp>

namespace redGrapes
{

namespace dispatch
    {
        namespace thread
        {
            using WorkerId = unsigned;
            struct WorkerPool;
        }
    }

extern std::shared_ptr< dispatch::thread::WorkerPool > worker_pool;

namespace memory
{
extern thread_local unsigned current_arena;

struct Allocator
{   
    dispatch::thread::WorkerId worker_id;

    Allocator() :Allocator(current_arena) {
                
            }
    Allocator( dispatch::thread::WorkerId worker_id );

    Block allocate( size_t n_bytes );
    void deallocate( Block blk );
};

template < typename T >
struct StdAllocator
{
    Allocator alloc;
    typedef T value_type;

    StdAllocator () : alloc( current_arena )
    {
    }
    StdAllocator( dispatch::thread::WorkerId worker_id ) : alloc( worker_id ) {}

    template< typename U >
    constexpr StdAllocator(StdAllocator<U> const& other) noexcept
        : alloc( other.alloc )
    {
    }

    inline T* allocate( std::size_t n )
    {
        Block blk = alloc.allocate( sizeof(T) * n );
        SPDLOG_TRACE("allocate {},{},{}", (uintptr_t)ptr, n*sizeof(T), boost::core::demangle(typeid(T).name()));

        return (T*) blk.ptr;
    }

    inline void deallocate(T* p, std::size_t n = 0) noexcept
    {
        alloc.deallocate(Block { .ptr = (uintptr_t)p, .len = sizeof(T)*n });
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
bool operator==(StdAllocator<T> const&, StdAllocator<U> const &) { return true; }
 
template<typename T, typename U>
bool operator!=(StdAllocator<T> const&, StdAllocator<U> const&) { return false; }


/* allocates a shared_ptr in the memory pool of a given worker
 */
template < typename T, typename... Args >
std::shared_ptr<T> alloc_shared_bind( dispatch::thread::WorkerId worker_id, Args&&... args )
{
    return std::allocate_shared< T, StdAllocator<T> >( StdAllocator<T>( worker_id ), std::forward<Args>(args)... );
}

/* allocates a shared_ptr in the memory pool of the current worker
 */
template < typename T, typename... Args >
std::shared_ptr<T> alloc_shared( Args&&... args )
{
    return std::allocate_shared< T, StdAllocator<T> >( StdAllocator<T>(), std::forward<Args>(args)... );
}

} // namespace memory
} // namespace redGrapes


