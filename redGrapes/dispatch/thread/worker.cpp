/* Copyright 2020-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/dispatch/thread/worker_pool.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

WorkerThread::WorkerThread( WorkerId worker_id )
    : id( worker_id ),
      thread(
             [this] {
                 /* setup membind- & cpubind policies using hwloc
                  */
                 this->cpubind();
                 this->membind();

                 /* since we are in a worker, there should always
                  * be a task running (we always have a parent task
                  * and therefore yield() guarantees to do
                  * a context-switch instead of idling
                  */
                 redGrapes::idle = [this] {
                     throw std::runtime_error("idle in worker thread!");
                 };

                 /* wait for start-flag to be triggerd in order
                  * to avoid premature access to `shared_from_this`
                  */
                 while( ! m_start.load(std::memory_order_consume) )
                     cv.wait();

                 /* initialize thread-local variables
                  */
                 current_worker = this->shared_from_this();
                 current_waker_id = this->get_waker_id();
                 memory::current_arena = this->get_worker_id();

                 /* execute tasks until stop()
                  */
                 this->work_loop();

                 SPDLOG_TRACE("Worker Finished!");
             }
             )
{    
}

void WorkerThread::start()
{
    m_start.store(true, std::memory_order_release);
    wake();
}

void WorkerThread::stop()
{
    SPDLOG_TRACE("Worker::stop()");
    m_stop.store(true, std::memory_order_release);
    wake();
    thread.join();
}

void WorkerThread::cpubind()
{
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, this->id);

    if( hwloc_set_cpubind(topology, obj->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) )
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
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, this->id);

    if( hwloc_set_membind(topology, obj->cpuset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT ) )
    {
        char *str;
        int error = errno;
        hwloc_bitmap_asprintf(&str, obj->cpuset);
        spdlog::warn("Couldn't membind to cpuset {}: {}\n", str, strerror(error));
        free(str);
    }    
}

void WorkerThread::work_loop()
{
    SPDLOG_TRACE("Worker {} start work_loop()", id);
    while( ! m_stop.load(std::memory_order_consume) )
    {
        while( Task * task = this->gather_task() )
        {
            worker_pool->set_worker_state( id, dispatch::thread::WorkerState::BUSY );
            dispatch::thread::execute_task( *task );
        }

        worker_pool->set_worker_state( id, dispatch::thread::WorkerState::BUSY );
            
        if( !m_stop.load(std::memory_order_consume) )
            cv.wait();
    }
    SPDLOG_TRACE("Worker {} end work_loop()", id);
}

Task * WorkerThread::gather_task()
{
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
    worker_pool->set_worker_state( id, dispatch::thread::WorkerState::AVAILABLE );
        
#ifndef ENABLE_WORKSTEALING
#define ENABLE_WORKSTEALING 1
#endif
        
#if ENABLE_WORKSTEALING

    /* STAGE 3:
     *
     * after all tasks are workstealing
     */
    SPDLOG_TRACE("Worker {}: try to steal tasks", id);
    task = top_scheduler->steal_task( *this );
        
#endif

    return task;
}

bool WorkerThread::init_dependencies( Task* & t, bool claimed )
{
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

