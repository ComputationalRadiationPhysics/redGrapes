#pragma once

#include <memory>
#include <functional>

namespace redGrapes
{

struct Task;
struct TaskSpace;

namespace dispatch {
namespace thread{
struct WorkerPool;
}
}

namespace scheduler {
struct IScheduler;
}

/*! global context
 */
extern thread_local Task * current_task;
extern thread_local std::function< void () > idle;

extern std::shared_ptr< TaskSpace > top_space;
extern std::shared_ptr< scheduler::IScheduler > top_scheduler;
extern std::shared_ptr< dispatch::thread::WorkerPool > worker_pool;

unsigned scope_depth();

} // namespace redGrapes

