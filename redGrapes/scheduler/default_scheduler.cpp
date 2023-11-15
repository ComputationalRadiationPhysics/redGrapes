
#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>
#include <redGrapes/util/trace.hpp>
#include <spdlog/spdlog.h>
#include <redGrapes/redGrapes.hpp>

namespace redGrapes
{
namespace scheduler
{

DefaultScheduler::DefaultScheduler( )
{
}

void DefaultScheduler::idle()
{
    SPDLOG_TRACE("DefaultScheduler::idle()");

    /* the main thread shall not do any busy waiting
     * and always sleep right away in order to
     * not block any worker threads (those however should
     * busy-wait to improve latency)
     */
    cv.timeout = 0;             
    cv.wait();
}

/* send the new task to a worker
 */
void DefaultScheduler::emplace_task( Task & task )
{
    // todo: properly store affinity information in task
    dispatch::thread::WorkerId worker_id = task.arena_id % SingletonContext::get().worker_pool->size();

    SingletonContext::get().worker_pool->get_worker(worker_id).emplace_task( task );
}

/* send this already existing task to a worker,
 * but only through follower-list so it is not assigned to a worker yet.
 * since this task is now ready, send find a worker for it
 */
void DefaultScheduler::activate_task( Task & task )
{
    //! worker id to use in case all workers are busy
    static thread_local std::atomic< unsigned int > next_worker(SingletonContext::get().current_worker ?
                                                                 SingletonContext::get().current_worker->get_worker_id() + 1 : 0);
    TRACE_EVENT("Scheduler", "activate_task");
    SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);

    int worker_id = SingletonContext::get().worker_pool->find_free_worker();
    if( worker_id < 0 )
    {
        worker_id = next_worker.fetch_add(1) % SingletonContext::get().worker_pool->size();
        if( worker_id == SingletonContext::get().current_worker->get_worker_id() )
            worker_id = next_worker.fetch_add(1) % SingletonContext::get().worker_pool->size();
    }

    SingletonContext::get().worker_pool->get_worker( worker_id ).ready_queue.push(&task);
    SingletonContext::get().worker_pool->set_worker_state( worker_id, dispatch::thread::WorkerState::BUSY );
    SingletonContext::get().worker_pool->get_worker( worker_id ).wake();
}

/* tries to find a task with uninialized dependency edges in the
 * task-graph in the emplacement queues of other workers
 * and removes it from there
 */
Task * DefaultScheduler::steal_new_task( dispatch::thread::Worker & worker )
{
    std::optional<Task*> task = SingletonContext::get().worker_pool->probe_worker_by_state<Task*>(
        [&worker](unsigned idx) -> std::optional<Task*>
        {
            // we have a candidate of a busy worker,
            // now check its queue
            if(Task* t = SingletonContext::get().worker_pool->get_worker(idx).emplacement_queue.pop())
                return t;

            // otherwise check own queue again
                else if(Task* t = worker.emplacement_queue.pop())
                    return t;

                // else continue search
                else
                    return std::nullopt;
            },

            // find a busy worker
            dispatch::thread::WorkerState::BUSY,

            // start next to current worker
            worker.get_worker_id());

        return task ? *task : nullptr;
    }

    /* tries to find a ready task in any queue of other workers
     * and removes it from the queue
     */
    Task * DefaultScheduler::steal_ready_task( dispatch::thread::Worker & worker )
    {
        std::optional<Task*> task = SingletonContext::get().worker_pool->probe_worker_by_state<Task*>(
            [&worker](unsigned idx) -> std::optional<Task*>
            {
                // we have a candidate of a busy worker,
                // now check its queue
                if(Task* t = SingletonContext::get().worker_pool->get_worker(idx).ready_queue.pop())
                    return t;

                // otherwise check own queue again
                else if(Task* t = worker.ready_queue.pop())
                    return t;

                // else continue search
                else
                    return std::nullopt;
            },

            // find a busy worker
            dispatch::thread::WorkerState::BUSY,

            // start next to current worker
            worker.get_worker_id());

        return task ? *task : nullptr;
    }

    // give worker a ready task if available
    // @return task if a new task was found, nullptr otherwise
    Task * DefaultScheduler::steal_task( dispatch::thread::Worker & worker )
    {
        unsigned worker_id = worker.get_worker_id();

        SPDLOG_INFO("steal task for worker {}", worker_id);

        if( Task * task = steal_ready_task( worker ) )
        {
            SingletonContext::get().worker_pool->set_worker_state( worker_id, dispatch::thread::WorkerState::BUSY );
            return task;
        }

        if( Task * task = steal_new_task( worker ) )
        {
            task->pre_event.up();
            task->init_graph();

            if( task->get_pre_event().notify( true ) )
            {
                SingletonContext::get().worker_pool->set_worker_state( worker_id, dispatch::thread::WorkerState::BUSY );
                return task;
            }            
        }

        return nullptr;
    }

    /* Wakeup some worker or the main thread
     *
     * WakerId = 0 for main thread
     * WakerId = WorkerId + 1
     *
     * @return true if thread was indeed asleep
     */
    bool DefaultScheduler::wake( WakerId id )
    {
        if( id == 0 )
            return cv.notify();
        else if( id > 0 && id <= SingletonContext::get().worker_pool->size() )
            return SingletonContext::get().worker_pool->get_worker(id - 1).wake();
        else
            return false;
    }

    /* wakeup all wakers (workers + main thread)
     */
    void DefaultScheduler::wake_all()
    {
        for( uint16_t i = 0; i <= SingletonContext::get().worker_pool->size(); ++i )
            this->wake( i );
    }

} // namespace scheduler
} // namespace redGrapes

