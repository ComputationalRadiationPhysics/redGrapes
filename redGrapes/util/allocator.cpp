#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>

#include <redGrapes/util/allocator.hpp>

namespace redGrapes
{
namespace memory
{

UntypedAllocator::UntypedAllocator( dispatch::thread::WorkerId worker_id )
  : worker_id( worker_id )
{}

void * UntypedAllocator::allocate( size_t n_bytes )
{
    return (void*)worker_pool->get_worker( worker_id ).alloc.allocate< uint8_t >( n_bytes );
}

void UntypedAllocator::deallocate( void * ptr )
{
    worker_pool->get_worker( worker_id ).alloc.deallocate( ptr );
}

} // namespace memory
} // namespace redGrapes

