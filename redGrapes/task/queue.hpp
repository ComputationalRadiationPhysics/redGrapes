/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <redGrapes/task/task.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <moodycamel/concurrentqueue.h>
#include <redGrapes/util/trace.hpp>

namespace redGrapes
{
namespace task
{

// Default traits for the ConcurrentQueue. To change some of the
// traits without re-implementing all of them, inherit from this
// struct and shadow the declarations you wish to be different;
// since the traits are used as a template type parameter, the
// shadowed declarations will be used where defined, and the defaults
// otherwise.
struct TaskQueueTraits
{
	// General-purpose size type. std::size_t is strongly recommended.
	typedef std::size_t size_t;
	
	// The type used for the enqueue and dequeue indices. Must be at least as
	// large as size_t. Should be significantly larger than the number of elements
	// you expect to hold at once, especially if you have a high turnover rate;
	// for example, on 32-bit x86, if you expect to have over a hundred million
	// elements or pump several million elements through your queue in a very
	// short space of time, using a 32-bit type *may* trigger a race condition.
	// A 64-bit int type is recommended in that case, and in practice will
	// prevent a race condition no matter the usage of the queue. Note that
	// whether the queue is lock-free with a 64-int type depends on the whether
	// std::atomic<std::uint64_t> is lock-free, which is platform-specific.
	typedef std::size_t index_t;
	
	// Internally, all elements are enqueued and dequeued from multi-element
	// blocks; this is the smallest controllable unit. If you expect few elements
	// but many producers, a smaller block size should be favoured. For few producers
	// and/or many elements, a larger block size is preferred. A sane default
	// is provided. Must be a power of 2.
	static const size_t BLOCK_SIZE = 64;

	// For explicit producers (i.e. when using a producer token), the block is
	// checked for being empty by iterating through a list of flags, one per element.
	// For large block sizes, this is too inefficient, and switching to an atomic
	// counter-based approach is faster. The switch is made for block sizes strictly
	// larger than this threshold.
	static const size_t EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD = 32;

	// How many full blocks can be expected for a single explicit producer? This should
	// reflect that number's maximum for optimal performance. Must be a power of 2.
	static const size_t EXPLICIT_INITIAL_INDEX_SIZE = 128;
	
	// How many full blocks can be expected for a single implicit producer? This should
	// reflect that number's maximum for optimal performance. Must be a power of 2.
	static const size_t IMPLICIT_INITIAL_INDEX_SIZE = 128;
	
	// The initial size of the hash table mapping thread IDs to implicit producers.
	// Note that the hash is resized every time it becomes half full.
	// Must be a power of two, and either 0 or at least 1. If 0, implicit production
	// (using the enqueue methods without an explicit producer token) is disabled.
	static const size_t INITIAL_IMPLICIT_PRODUCER_HASH_SIZE = 32;
	
	// Controls the number of items that an explicit consumer (i.e. one with a token)
	// must consume before it causes all consumers to rotate and move on to the next
	// internal queue.
	static const std::uint32_t EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE = 256;
	
	// The maximum number of elements (inclusive) that can be enqueued to a sub-queue.
	// Enqueue operations that would cause this limit to be surpassed will fail. Note
	// that this limit is enforced at the block level (for performance reasons), i.e.
	// it's rounded up to the nearest block size.
  static const size_t MAX_SUBQUEUE_SIZE = moodycamel::details::const_numeric_max<size_t>::value;

	// The number of times to spin before sleeping when waiting on a semaphore.
	// Recommended values are on the order of 1000-10000 unless the number of
	// consumer threads exceeds the number of idle cores (in which case try 0-100).
	// Only affects instances of the BlockingConcurrentQueue.
	static const int MAX_SEMA_SPINS = 10000;

	// Whether to recycle dynamically-allocated blocks into an internal free list or
	// not. If false, only pre-allocated blocks (controlled by the constructor
	// arguments) will be recycled, and all others will be `free`d back to the heap.
	// Note that blocks consumed by explicit producers are only freed on destruction
	// of the queue (not following destruction of the token) regardless of this trait.
	static const bool RECYCLE_ALLOCATED_BLOCKS = false;

	static inline void* malloc(size_t size) {
                return std::malloc(size);
//      return (void*) memory::Allocator< uint8_t >().allocate( size );
  }
	static inline void free(void* ptr) {
                std::free( ptr );
//      memory::Allocator< uint8_t >().deallocate( (uint8_t*)ptr, 1 );
  }
};

struct Queue
{
    /*
    std::atomic< Task * > head;
    std::atomic< Task * > tail;

    std::mutex m;
*/
    moodycamel::ConcurrentQueue< Task*/*, TaskQueueTraits */> cq;

    Queue();
    Queue( unsigned capacity ) :cq(capacity) {
    }

    inline void push(Task * task)
    {
        TRACE_EVENT("Task", "TaskQueue::push()");
        this->cq.enqueue(task);
    }

    inline Task * pop()
    {
        TRACE_EVENT("Task", "TaskQueue::pop()");
        Task * t = nullptr;
        if( this->cq.try_dequeue( t ) )
            return t;
        else
            return nullptr;
    }
};

}
}
