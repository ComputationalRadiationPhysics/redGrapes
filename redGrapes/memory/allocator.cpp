#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>

#include <redGrapes/memory/allocator.hpp>

namespace redGrapes
{
namespace memory
{

UntypedAllocator::UntypedAllocator( dispatch::thread::WorkerId worker_id )
  : worker_id( worker_id )
{}

void * UntypedAllocator::allocate( size_t n_bytes )
{
    return (void*)worker_pool->get_alloc( worker_id ).allocate< uint8_t >( n_bytes );
}

void UntypedAllocator::deallocate( void * ptr )
{
    worker_pool->get_alloc( worker_id ).deallocate( ptr );
}

} // namespace memory
} // namespace redGrapes

