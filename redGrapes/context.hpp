#pragma once

#include <memory>
#include <moodycamel/concurrentqueue.h>

#include <redGrapes/scheduler/scheduler.hpp>

namespace redGrapes
{

struct Task;
struct TaskSpace;

/*! global context
 */
extern moodycamel::ConcurrentQueue< std::shared_ptr<TaskSpace> > active_task_spaces;
extern std::shared_ptr< TaskSpace > top_space;
extern std::shared_ptr< scheduler::IScheduler > top_scheduler;

extern thread_local std::shared_ptr<Task> current_task;
extern thread_local std::function< void () > idle;

unsigned scope_depth();

} // namespace redGrapes

