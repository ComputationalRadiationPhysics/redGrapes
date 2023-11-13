#include <memory>
#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>

#include <redGrapes/memory/allocator.hpp>

namespace redGrapes
{
namespace memory
{

Allocator::Allocator( dispatch::thread::WorkerId worker_id )
  : worker_id( worker_id )
{}

Block Allocator::allocate( size_t n_bytes )
{
    return worker_pool->get_alloc( worker_id ).allocate( n_bytes );
}

void Allocator::deallocate( Block blk )
{
    worker_pool->get_alloc( worker_id ).deallocate( blk );
}

} // namespace memory
} // namespace redGrapes

