/* Copyright 2020-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <atomic>
#include <hwloc.h>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/memory/hwloc_alloc.hpp>
#include <redGrapes/memory/chunked_bump_alloc.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/redGrapes.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{
WorkerThread::WorkerThread( memory::ChunkedBumpAlloc< memory::HwlocAlloc > & alloc, HwlocContext & hwloc_ctx, hwloc_obj_t const & obj, WorkerId worker_id )
    : Worker( alloc, hwloc_ctx, obj, worker_id )
{
}

WorkerThread::~WorkerThread()
{
}

void WorkerThread::start()
{
    thread = std::thread([this]{ this->run(); });
}

Worker::Worker( memory::ChunkedBumpAlloc<memory::HwlocAlloc> & alloc, HwlocContext & hwloc_ctx, hwloc_obj_t const & obj, WorkerId worker_id )
    : alloc( alloc )
    , hwloc_ctx( hwloc_ctx )
    , id( worker_id )
{
}

Worker::~Worker()
{
}

void Worker::stop()
{
    SPDLOG_TRACE("Worker::stop()");
    m_stop.store(true, std::memory_order_release);
    wake();
}

void WorkerThread::stop()
{
    Worker::stop();
    thread.join();
}

void WorkerThread::run()
{
    /* setup membind- & cpubind policies using hwloc
     */
    this->cpubind();
    this->membind();

    /* since we are in a worker, there should always
     * be a task running (we always have a parent task
     * and therefore yield() guarantees to do
     * a context-switch instead of idling
     */
                /*
    idle = [this] {
        throw std::runtime_error("idle in worker thread!");
    };
                */

    /* initialize thread-local variables
     */
    SingletonContext::get().current_worker = this->shared_from_this();
    SingletonContext::get().current_waker_id = this->get_waker_id();
    SingletonContext::get().current_arena = this->get_worker_id();

    /* execute tasks until stop()
     */
    this->work_loop();

    SingletonContext::get().current_worker.reset();

    SPDLOG_TRACE("Worker Finished!");    
}

void WorkerThread::cpubind()
{
    size_t n_pus = hwloc_get_nbobjs_by_type(hwloc_ctx.topology, HWLOC_OBJ_PU);
    hwloc_obj_t obj = hwloc_get_obj_by_type(hwloc_ctx.topology, HWLOC_OBJ_PU, id%n_pus);

    if( hwloc_set_cpubind(hwloc_ctx.topology, obj->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) )
    {
        char *str;
        int error = errno;
        hwloc_bitmap_asprintf(&str, obj->cpuset);
        spdlog::warn("Couldn't cpubind to cpuset {}: {}\n", str, strerror(error));
        free(str);
    }
}

void WorkerThread::membind()
{
    size_t n_pus = hwloc_get_nbobjs_by_type(hwloc_ctx.topology, HWLOC_OBJ_PU);
    hwloc_obj_t obj = hwloc_get_obj_by_type(hwloc_ctx.topology, HWLOC_OBJ_PU, id%n_pus);
    if( hwloc_set_membind(hwloc_ctx.topology, obj->cpuset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT ) )
    {
        char *str;
        int error = errno;
        hwloc_bitmap_asprintf(&str, obj->cpuset);
        spdlog::warn("Couldn't membind to cpuset {}: {}\n", str, strerror(error));
        free(str);
    }
}

void Worker::work_loop()
{
    SPDLOG_TRACE("Worker {} start work_loop()", id);
    while( ! m_stop.load(std::memory_order_consume) )
    {        
        SingletonContext::get().worker_pool->set_worker_state( id, dispatch::thread::WorkerState::AVAILABLE );
        cv.wait();
        
        while( Task * task = this->gather_task() )
        {
            SingletonContext::get().worker_pool->set_worker_state( id, dispatch::thread::WorkerState::BUSY );
            SingletonContext::get().execute_task( *task );
        }

    }
    SPDLOG_TRACE("Worker {} end work_loop()", id);
}

Task * Worker::gather_task()
{
    TRACE_EVENT("Worker", "gather_task()");
    Task * task = nullptr;

    /* STAGE 1:
     *
     * first, execute all tasks in the ready queue
     */
    SPDLOG_TRACE("Worker {}: consume ready queue", id);
    if( task = ready_queue.pop() )
        return task;

    /* STAGE 2:
     *
     * after the ready queue is fully consumed,
     * try initializing new tasks until one
     * of them is found to be ready
     */
    SPDLOG_TRACE("Worker {}: try init new tasks", id);
    while( this->init_dependencies( task, true ) )
        if( task )
            return task;

    /* set worker state to signal that we are requesting tasks
     */
    SingletonContext::get().worker_pool->set_worker_state( id, dispatch::thread::WorkerState::AVAILABLE );
        
#ifndef ENABLE_WORKSTEALING
#define ENABLE_WORKSTEALING 1
#endif
        
#if ENABLE_WORKSTEALING

    /* STAGE 3:
     *
     * after all tasks from own queues are consumed, try to steal tasks
     */
    SPDLOG_TRACE("Worker {}: try to steal tasks", id);
    task = SingletonContext::get().scheduler->steal_task( *this );

#endif

    return task;
}

bool Worker::init_dependencies( Task* & t, bool claimed )
{
    TRACE_EVENT("Worker", "init_dependencies()");
    if(Task * task = emplacement_queue.pop())
    {
        SPDLOG_DEBUG("init task {}", task->task_id);

        task->pre_event.up();
        task->init_graph();

        if( task->get_pre_event().notify( claimed ) )
            t = task;
        else
        {
            t = nullptr;
        }

        return true;
    }
    else
        return false;
}

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

