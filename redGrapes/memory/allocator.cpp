#include <memory>
#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>

#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/redGrapes.hpp>

namespace redGrapes
{
namespace memory
{

Allocator::Allocator()
  : Allocator(SingletonContext::get().current_arena) {}

Allocator::Allocator( dispatch::thread::WorkerId worker_id )
  : worker_id(
          SingletonContext::get().n_workers == 0u ? 0u : worker_id % SingletonContext::get().n_workers
    )
{}

Block Allocator::allocate( size_t n_bytes )
{
    return SingletonContext::get().worker_pool->get_alloc( worker_id ).allocate( n_bytes );
}

void Allocator::deallocate( Block blk )
{
    SingletonContext::get().worker_pool->get_alloc( worker_id ).deallocate( blk );
}

} // namespace memory
} // namespace redGrapes

