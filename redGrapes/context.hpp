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
extern thread_local Task * current_task;
extern thread_local std::function< void () > idle;

extern std::shared_ptr< TaskSpace > top_space;
extern std::shared_ptr< scheduler::IScheduler > top_scheduler;

unsigned scope_depth();

} // namespace redGrapes

