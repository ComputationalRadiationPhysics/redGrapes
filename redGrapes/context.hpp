#pragma once

#include <memory>
#include <functional>

namespace redGrapes
{

struct Task;
struct TaskSpace;
struct HwlocContext;
 
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

extern thread_local unsigned next_worker;

//struct Context
 
//{
extern    std::shared_ptr< HwlocContext > hwloc_ctx;
extern    std::shared_ptr< dispatch::thread::WorkerPool > worker_pool;
extern    std::shared_ptr< TaskSpace > top_space;
extern    std::shared_ptr< scheduler::IScheduler > top_scheduler;
//};

//extern Context ctx;

unsigned scope_depth();

} // namespace redGrapes

